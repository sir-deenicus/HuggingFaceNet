namespace TensorNet

open System
open Tokenizers
open MathNet.Numerics.LinearAlgebra
open TensorExtensions
open System.Text
open System.Collections.Generic
open Newtonsoft.Json
open MathNet.Numerics

type DocumentItem = { Text: string; Title: string }

type SearchResult =
    { Text: string
      Similarity: float32
      Title: string }

module Documents =
    let toVectors (tokenizer: GeneralTokenizer) (model: ONNX.NNet<float32>) startIndex (docs: string [] []) =
        let mutable vectors = DenseMatrix.ofRowArrays [| [| 0.f |] |]
        let indices = ResizeArray<DocLoc>()

        for i in 0..100 .. docs.Length - 1 do
            if i % 1000 = 0 then
                printfn "Document # %d %A" i (float i / (float docs.Length - 1.))

                if i % 10_000 = 0 then
                    model.ResetMemory()

            let k = startIndex + i
            let defs = docs[i .. i + 99]

            let toks = tokenizer.BatchTokenize(defs, k)
            let result = model.RunT(toks.TokensAndMasks)

            let m = Tensor.toMatrix result[0]
            printfn "indices: %A | matrix row count: %A" toks.Indices.Length m.RowCount

            indices.AddRange toks.Indices

            model.GC()
            GC.Collect()

            match vectors.RowCount, vectors.ColumnCount with
            | (1, 1) -> vectors <- m
            | _ -> vectors <- vectors.Stack m

        vectors, indices

//Interface has get doc and filterToNewDocs:
type IDocumentStore =
    abstract member GetDoc: int -> DocumentItem
    abstract member Documents: ResizeArray<DocumentItem>
    abstract member FilterToNewDocs: DocumentItem [] -> DocumentItem []
    abstract member Vectors: Matrix<float32> option
    abstract member RandomProjectionMatrix: Matrix<float32> option
    abstract member Indices: ResizeArray<DocLoc>
    abstract member Process: DocumentItem [] * Matrix<float32> * DocLoc ResizeArray -> unit
    abstract member Save: unit -> unit
    abstract member Load: unit -> unit


//A class that stores the document vectors and document indices.
//Should allow updating and checking if file exists
//implements IDocumentStore

type DocumentVectorState =
    { mutable Documents: DocumentItem ResizeArray
      mutable DocHashes: uint64 HashSet
      mutable Indices: DocLoc ResizeArray
      mutable Vectors: Matrix<float32> option
      mutable RandomProjectionMatrix: Matrix<float32> option }

    //default
    static member New() =
        { Documents = ResizeArray<DocumentItem>()
          DocHashes = HashSet<uint64>()
          Indices = ResizeArray<DocLoc>()
          Vectors = None
          RandomProjectionMatrix = None }

type DocumentVectorStore(storeloc: string, fname: string, ?vectorStateInit) =
    let mutable vectorState =
        defaultArg
            vectorStateInit
            { Documents = ResizeArray()
              DocHashes = HashSet()
              Indices = ResizeArray()
              Vectors = None
              RandomProjectionMatrix = None }


    let hasher (s: string) =
        let bytes = System.Text.Encoding.UTF8.GetBytes(s)
        K4os.Hash.xxHash.XXH64.DigestOf(bytes)

    member __.Save() =
        //serialize docs to json:
        let json = JsonConvert.SerializeObject(vectorState.Documents)
        IO.File.WriteAllText(IO.Path.Combine(storeloc, fname + ".json"), json)

        Files.save (IO.Path.Combine(storeloc, fname + ".bin")) vectorState


    member __.Load() =
        vectorState <- Files.load (IO.Path.Combine(storeloc, fname + ".bin")) 

        let json = IO.File.ReadAllText(IO.Path.Combine(storeloc, fname + ".json"))
        vectorState.Documents <- JsonConvert.DeserializeObject<ResizeArray<DocumentItem>>(json)

    member __.FilterToNewDocuments(docs: DocumentItem []) =
        docs
        |> Array.filter (fun d -> not (vectorState.DocHashes.Contains(hasher d.Text)))

    member __.ProcessDocuments(newdocs: DocumentItem [], newvectors: Matrix<float32>, newindices: DocLoc ResizeArray) =

        for doc in newdocs do
            vectorState.Documents.Add(doc)

            vectorState.DocHashes.Add(hasher doc.Text)
            |> ignore

        //We do not check if docs is already contained why?
        //Answer: because we are only adding new docs to the store and we know they are new because
        //we checked the hashset before in FilterToNewDocuments before calling this method.

        //fix is to rename vectors to vectorState.Vectors. Apply:
        match vectorState.Vectors with
        | Some v -> vectorState.Vectors <- Some(v.Stack newvectors)
        | None -> vectorState.Vectors <- Some newvectors

        vectorState.Indices.AddRange newindices

    member __.ApplyRandomProjection(newDim) =
        let randprojmatrix =
            vectorState.Vectors
            |> Option.map (fun v -> Matrix<float32>.Build.Random (v.ColumnCount, newDim, Distributions.Normal(0., 1.)))

        let newVectorState =
            { Documents = ResizeArray(vectorState.Documents)
              DocHashes = HashSet(vectorState.DocHashes)
              Indices = ResizeArray(vectorState.Indices)
              Vectors =
                (vectorState.Vectors, randprojmatrix)
                ||> Option.map2 (fun v m ->
                    let rm = v * m
                    //normalize rows
                    rm.NormalizeRows(2.))
              RandomProjectionMatrix = randprojmatrix }

        DocumentVectorStore(storeloc, fname + "-rp", newVectorState)


    //implement interface
    interface IDocumentStore with
        member __.Vectors = vectorState.Vectors

        member __.Indices = vectorState.Indices

        member __.Documents = vectorState.Documents

        member __.RandomProjectionMatrix = vectorState.RandomProjectionMatrix

        member __.GetDoc(i: int) = vectorState.Documents.[i]

        member this.Process(newdocs, newvectors, newindices) =
            this.ProcessDocuments(newdocs, newvectors, newindices)

        member this.FilterToNewDocs(docs: DocumentItem []) = this.FilterToNewDocuments(docs)

        member this.Save() = this.Save()

        member this.Load() = this.Load()



type SentenceVectors(tokenizer: GeneralTokenizer, model: ONNX.NNet<float32>, documents: IDocumentStore, ?defaultSimilarityThreshold) =
    let defaultSimilarityThreshold = defaultArg defaultSimilarityThreshold 0.45f
    
    //compute intersection of N arrays where each item has associated score, take score averages

    let conceptsIntersection threshold (xs: _ [] []) =
        if xs.Length > 0 then
            //find shortest in xs
            let shortest = xs |> Array.minBy (fun x -> x.Length)

            //partition into shortest and rest
            let (shortests, rest) =
                xs
                |> Array.partition (fun x -> x.Length = shortest.Length)

            let shortest = shortests[0]
            //Question: do you know why we partitioned the array and picked an element?
            //Answer: Because there might **be multiple shortest arrays, and we want to take the first one**

            //append everything after 1st elem to rest

            //let rest = Array.append shortests[1..] rest
            //there is a hazard in the line above. What is it?
            //Answer: **If there are no shortests, then shortests[1..] will throw an exception**
            //Actually not just the em**pty array, but any array with length < 2 will throw an exception**
            //**So we need to check if shortests has length > 1**
            //let's fix this
            let rest =
                if shortests.Length > 1 then
                    Array.append shortests[1..] rest
                else
                    rest

            //map rest to restsets of dicts
            let restdict = Array.map dict rest

            //create array, iterate through shortest, add only if in each of the rest sets
            [| for (item, score: float32) in shortest do
                   let containedInAll =
                       restdict
                       |> Array.forall (fun rest -> rest.ContainsKey item)

                   if containedInAll then
                       //take average of scores, including score
                       let restscoreSum = restdict |> Array.sumBy (fun rest -> rest[item])

                       let scoreavg =
                           (score + restscoreSum)
                           / (float32 restdict.Length + 1f)
                       //do you know why there is a +1 there?
                       //**Answer: Because we are including the score of the shortest array**

                       if scoreavg > threshold then
                           yield item, scoreavg |]
            |> Array.sortByDescending snd
        else
            [||]

    let getNeighborVectors (str: string) =
        match documents.Vectors with
        | Some vecs ->
            let v =
                model.RunT(tokenizer.Tokenize str)
                |> Array.head
                |> Tensor.toVector

            //vecs * v |> Seq.toArray
            //hmm vecs might be random projected, so we need to multiply by random projection matrix
            //let's fix this
            match documents.RandomProjectionMatrix with
            | Some m -> vecs * (v * m).Normalize(2.) |> Seq.toArray
            | None -> vecs * v |> Seq.toArray

        | None -> [||]


    member __.Vectors = documents.Vectors

    member __.DocumentStore = documents

    member __.Dimension = model.OutputMetaData

    member __.Process(docs: DocumentItem []) =
        let newdocs = documents.FilterToNewDocs docs
        printfn $"{newdocs.Length} new documents"

        let vecs, indices =
            newdocs
            //|> Array.map splitBySentences
            //above Array.map is wrong cause it assumes string type fix to use document item type, use just .Text
            |> Array.map (fun d -> d.Text |> splitBySentences)
            |> Documents.toVectors tokenizer model documents.Indices.Count

        documents.Process(newdocs, vecs, indices)

    //function to tokenize, run input and multiply matrix
    member __.GetNeighborVectors(str: string) = getNeighborVectors str

    member __.GetNeighbors(str: string, ?minthresh, ?useLenHeuristic) =

        let minthresh = defaultArg minthresh defaultSimilarityThreshold
        let useHeuristic = defaultArg useLenHeuristic true

        let vs = getNeighborVectors str

        [| for i in 0 .. vs.Length - 1 do
               let sim = vs[i]

               if sim > minthresh then
                   let j = documents.Indices[i].GlobalLoc
                   let docitem = documents.GetDoc j
                   let txtlen = docitem.Text.Split(' ').Length

                   let str =
                       if useHeuristic
                          && tokenizer.MaxContextWindowLen > txtlen * 2 then
                           docitem.Text
                       else
                           let k = documents.Indices[i].InnerLoc
                           tokenizer.SplitToTokenizerWindow(docitem.Text).[k]

                   { Title = docitem.Title
                     Text = str
                     Similarity = sim } |]
        |> Array.sortByDescending (fun d -> d.Similarity)


    member __.Intersection(strs: string seq, ?threshold: float32, ?useLenHeuristic) =
        let threshold = defaultArg threshold defaultSimilarityThreshold
        let useLenHeuristic = defaultArg useLenHeuristic true

        let space =
            [| for str in strs do
                   let vs = getNeighborVectors str

                   let txts =
                       [| for i in 0 .. vs.Length - 1 do
                              let sim = vs[i]

                              if sim > threshold then
                                  let doc = documents.GetDoc documents.Indices.[i].GlobalLoc
                                  let j = documents.Indices.[i].InnerLoc

                                  let txt =
                                      if useLenHeuristic
                                         && tokenizer.MaxContextWindowLen > doc.Text.Split(' ').Length * 2 then
                                          doc.Text
                                      else
                                          tokenizer.SplitToTokenizerWindow(doc.Text).[j]

                                  txt, vs.[i] |]

                   txts |]

        conceptsIntersection threshold space
        |> Array.sortByDescending snd

    //parameter array overload of Intersection
    member __.Intersection([<ParamArray>] strs: string []) = __.Intersection(strs)
