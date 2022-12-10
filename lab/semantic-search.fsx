#I @"C:\Users\cybernetic\.nuget\packages\"
#I @"D:\Downloads\NeuralNets\onnx1.13"
#I @"C:\Users\cybernetic\source\repos\"
#r @"BlingFire\bin\Release\net5.0\BlingFire.dll"
#r @"Prelude\Prelude\bin\Release\net5\Prelude.dll"
#r @"..\bin\x64\Debug\net5.0\TensorNet.dll"
#r @"mathnet.numerics\5.0.0\lib\net5.0\MathNet.Numerics.dll"
#r @"mathnet.numerics.fsharp\5.0.0\lib\net5.0\MathNet.Numerics.FSharp.dll"
#r @"D:\Downloads\NeuralNets\onnx1.13\Microsoft.ML.OnnxRuntime.dll"

#r @"newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"
#r "nuget: K4os.Hash.xxHash, 1.0.8"
#r "nuget: FsPickler, 5.3.2"
#r @"C:\Users\cybernetic\source\repos\SentencePieceWrapper\SentencePieceDotNET\bin\x64\Release\net5.0\SentencePieceDotNET.dll" 

#time "on"

open TensorNet 
open TensorExtensions
open System
open Prelude.Common
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.MatrixExtensions
open Newtonsoft.Json
open TensorNet.Tokenizers

let model =
    new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\deberta-v3-nli-base\deberta-v3-nli-base.onnx")
    
    
let model =
    new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2.onnx")

let model2 =
    new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2-opt.onnx")

let model3 =
    new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2-quantized.onnx")
let db = Tokenizers.SentencePieceTokenizer.NewDebertaTokenizer(@"D:\Downloads\NeuralNets\deberta-v3-nli-base\spm.model")

    
Tokenizers.initBlingFire @"D:\Downloads\NeuralNets\blingfire\blingfiretokdll.dll"


let flant5smallEncoder = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\flan-t5-small\flan-t5-small-encoder.onnx")
let flant5smallDecoder = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\flan-t5-small\flan-t5-small-decoder.onnx")

let flant5tokenizer = Tokenizers.SentencePieceTokenizer.NewT5Tokenizer(@"D:\Downloads\NeuralNets\flan-t5-small\spiece.model")

let flant5small = new ONNX.EncoderDecoderSampler(flant5smallEncoder, flant5smallDecoder, PadTokens.T5, flant5tokenizer.StopToken, sentenceEndTokens = flant5tokenizer.EndOfSentenceTokens.Value)

    
let q = "Language models (LMs) are trained on collections of documents, written by individual human agents to achieve specific goals in an outside world. During training, LMs have access only to text of these documents, with no direct evidence of the internal states of the agents that produced them -- a fact often used to argue that LMs are incapable of modeling goal-directed aspects of human language production and comprehension. Can LMs trained on text learn anything at all about the relationship between language and use? I argue that LMs are models of intentional communication in a specific, narrow sense."
let instr = "Definition: In this task, you are given the abstract of a research paper. Your task is to describe the key question being studied. Input: " + q + ". Output: "//"Definition: Create a bullet point list of the 4 main ideas covered in the text . Input: " + q + " Output: "
let q = """Machine learning systems perform well on pattern matching tasks, but their ability to perform algorithmic or logical reasoning is not well understood. One important reasoning capability is algorithmic extrapolation, in which models trained only on small/simple reasoning problems can synthesize complex strategies for large/complex problems at test time. Algorithmic extrapolation can be achieved through recurrent systems, which can be iterated many times to solve difficult reasoning problems. We observe that this approach fails to scale to highly complex problems because behavior degenerates when many iterations are applied -- an issue we refer to as "overthinking." We propose a recall architecture that keeps an explicit copy of the problem instance in memory so that it cannot be forgotten. We also employ a progressive training routine that prevents the model from learning behaviors that are specific to iteration number and instead pushes it to learn behaviors that can be repeated indefinitely. These innovations prevent the overthinking problem, and enable recurrent systems to solve extremely hard extrapolation tasks."""
let instr = "Definition: In this task, you must identify and list all of the key concepts of the text. The concepts must consist of more than one word. For example 'machine learning' is correct and 'machine','learning' is wrong. Input: " + q + ". Output: "
let instr = "Definition: In this task, you must identify and list all of the subjects of the text. Input: " + q + ". Output: " // *
let instr = "Definition: In this task, you must identify and list all of the concepts of the text. Input: " + q + ". Output: "


let instr = "Definition: For this task two steps are to be performed. Summarize the given text into a title that captures the main concept in the text. The title should be long. Input: " + q + ". Output: "
 
let r1 = flant5tokenizer.RawTokenize("Definition: Write a five line poem about going to the shop. Poem: ")//q + ". In summary: ") //
r1.Length

let sample k p samplingInfo samplerState (logits: float32 []) = 
    if samplerState.SampleLen >= samplingInfo.BlockStartTokenIndex then
        logits.[samplingInfo.StartToken] <- -infinityf

    //suppress end of sentence tokens while < minsentlen
    if samplerState.SampleLen < samplingInfo.MinSentLen then
        logits.[samplingInfo.StopToken] <- -infinityf

    let probs =
        [| for i in 0 .. logits.Length - 1 -> i, exp (float32 logits.[i]) |]
        |> Array.normalize

    let choices = Sampling.getLargeProbItems p probs 
    printfn "%A" (choices.[..9])
    let i =
        if k > 0 then
            choices |> Array.take k |> Sampling.discreteSampleIndex
        else
            Sampling.discreteSampleIndex choices
 
    choices.[i]    

let sampleTypical k p samplingInfo samplerState (logits: float32 []) = 
    if samplerState.SampleLen >= samplingInfo.BlockStartTokenIndex then
        logits.[samplingInfo.StartToken] <- -infinityf

    //suppress end of sentence tokens while < minsentlen
    if samplerState.SampleLen < samplingInfo.MinSentLen then
        logits.[samplingInfo.StopToken] <- -infinityf

    let probs =
        [| for i in 0 .. logits.Length - 1 -> i, exp (float32 logits.[i]) |]
        |> Array.normalize

    //what does the above line of code do?
    //Answer: it normalizes the probabilities so that they sum to 1

    let ent = -1f * (Array.sumBy (fun (_, p) -> p * log (p + 1e-30f)) probs)
    //what does the above line of code do?
    //Answer: it calculates the entropy of the probabilities
        
    let sorted = probs |> Array.sortBy (fun (_, p) -> abs (-log p - ent))  
    //printfn "%A" sm
    //what does the above line of code do?
    //Answer: it sorts the probabilities by the absolute value of the difference between the log of the probability and the entropy of the probabilities (i.e. it sorts the probabilities by how different they are from the average probability)
     
    let choices = 
        if k = 0 then Sampling.getTopProbs p sorted 
        else sorted |> Array.takeOrMax k |> Sampling.getTopProbs p
    printfn "%A" (choices)
    let i = Sampling.discreteSampleIndex choices

    choices[i]

        
let res = flant5small.Run(r1, countBySentences = false, minlen = 0, sampler = sampleTypical 0 0.1f)
res.Tokens.Length
flant5tokenizer.Detokenize(res.Tokens)

let r1 = flant5tokenizer.Tokenize("Definition: Write a five line poem about going to the shop. Poem: ")

flant5small.RunEncoder(r1) //([|0; 5879|])

flant5small.RunDecoderStep([|0|])

//make it int65
let r = db.RawTokenize ("hello, this is a test")
//ATTENTION MASK of just 1s for r
let attn = r |> Array.map (fun _ -> 1L)
//zip r and attn with input keys


//how do I remove a key from a python dictionary?
//Answer: del dict[key]

{db.Tokenize("hello, this is a test") with TokenIds = r.TokenIds |> Seq.map int64}

model.RunT(db.Tokenize("hello, this is a test"))

let bertTokenizer =
    Tokenizers.BlingFireTokenizer.NewBertTokenizer(@"D:\Downloads\NeuralNets\blingfire\")

let docstore = TensorNet.DocumentVectorStore(DocumentsFolder, "wordnetdb")

let sv = TensorNet.SentenceVectors(bertTokenizer, model3, docstore)

sv.DocumentStore.Load()

let docstoreRP = docstore.ApplyRandomProjection(120)
//let docstoreRP = TensorNet.DocumentVectorStore(DocumentsFolder, "wordnetdb-rp")
let svrp = TensorNet.SentenceVectors(bertTokenizer, model3, docstoreRP)
svrp.DocumentStore.Load()
svrp.Vectors.Value.ColumnCount


let corpus =
    [| "A man is eating food."
       "A man is eating a piece of bread."
       "The girl is carrying a baby."
       "A man is riding a horse."
       "A woman is playing violin."
       "Two men pushed carts through the woods."
       "A man is riding a white horse on an enclosed ground."
       "A monkey is playing drums."
       "the cat went out and about"
       "A cheetah is running behind its prey."
       "Crystallographers will be presenting their novel research"
       "The felines are playing"
       "We made a scientific discovery on the nature of the minerals"
       "We found some interesting new facts about these rocks" |]

let t = bertTokenizer.BatchTokenize(corpus)

t.TokensAndMasks.AttentionMasks
|> Tensor.toArray2D

t.TokensAndMasks.TokenIds |> Tensor.toArray2D

let res = model.RunT(t.TokensAndMasks)
let res2 = model2.RunT(t.TokensAndMasks)
let res3 = model3.RunT(t.TokensAndMasks)

res[0].Shape

let m = res[0] |> Tensor.toMatrix
let m2 = res2[0] |> Tensor.toMatrix
let m3 = res3[0] |> Tensor.toMatrix


m[0, *] * m[1, *]

m[ 0, * ].Norm(2.)

let sim = m * m.Transpose()
m2 * m2.Transpose()
m3 * m3.Transpose()

(m - m3) |> Matrix.Abs |> Matrix.sum

(m - m3)
|> Matrix.Abs
|> Matrix.toRowArrays
|> Array.map Array.average

(m - m3)
|> Matrix.Abs
|> Matrix.toRowArrays
|> Array.concat
|> Array.average


Array.zip corpus (sim[ 12, * ].ToArray())
|> Array.sortByDescending snd


sv.Process(
    corpus
    |> Array.map (fun s -> { Title = ""; Text = s })
)

sv.Vectors

sv.GetNeighbors("rocs")

sv.Intersection([| "food"; "minerals" |], 0.35f)

//--------

let lookup =
    IO.File.ReadAllText(pathCombine DocumentsFolder "wordnet-text.json")
    |> Newtonsoft.Json.JsonConvert.DeserializeObject<string []>

//let's time the next bit of code
let sw = System.Diagnostics.Stopwatch() 
sw.Start()

sv.Process(
    lookup
    |> Array.map (fun s -> { Title = s.Split(':')[0]; Text = s })
)

sw.Stop()
printfn "Elapsed time: %A" sw.Elapsed

sv.DocumentStore.Save()

sv.Vectors.Value.RowCount
sv.DocumentStore.Indices.Count
sv.DocumentStore.Documents.Count
 
svrp.DocumentStore.RandomProjectionMatrix.Value.ColumnCount 
 
let tests = ["kidney insulin resistance";"algorithm used for bayesian inference";"algorithm used by neural networks to learn";"approach to AI originally modeled after brains"; "graph for probability based reasoning"; "pki"; "gray matter"]
svrp.GetNeighbors(tests[6], 0.45f)
|> Array.indexed
|> Array.filter (fun (_, s) -> s.Text.ToLower().Contains("importance"))

sv.GetNeighbors("Prosopagnosia", 0.5f)//comedy routine where there are two people and one is silly and the other is serious")

|> Array.indexed
//start with c
//|> Array.filter (fun (_,s) -> s.Text.ToLower().StartsWith("str"))
//sort by title
|> Array.sortBy (fun (_,s) -> s.Title.Length)
|> Array.sortBy (fun (_,s) -> s.Text.Length)

svrp.DocumentStore.Save()
svrp.GetNeighbors("fmri", 0.5f)

log (200_000.) / (0.3 ** 2.)
//Question:If pareidolia represents the tendency for humans to find faces and other shapes in otherwise nebulous images, what is the word for the tendency for humans to assign a sense of agency or consciousness or sentience to an otherwise insentient artifact?
//Answer:Anthropomorphism
540 - 504

sv.Intersection([ "pareidolia";"Anthropomorphism" ], 0.25f)
//start with c
|> Array.filter (fun (s,_) -> s.StartsWith("c"))


DateTime.Parse("2:20 PM").AddHours(-5.).AddMinutes(-3.).AddHours(10.)
//Time: 12:09 PM. Target Time: 7:17PM. Time till target (Format: N hours and M minutes): 7 hours and 8 minutes

DateTime.Now - DateTime.Parse("2:20 PM") 

//we want to stream an xml file and extract the text from the <abstract> and <title> tags
//we want to do this in a streaming fashion, so we don't have to load the whole file into memory
//xml file is D:\Downloads\wikis\simplewiki-20220601-abstract.xml
//stream xml file
//extract text from abstract and title tags
//start

let xml =
    System.Xml.XmlReader.Create(@"D:\Downloads\wikis\simplewiki-20220601-abstract.xml")

let sw = System.Diagnostics.Stopwatch()
sw.Start()
//5. / 22. = 0.22727272727272727
5./22.
let rec loop (xml: System.Xml.XmlReader) = seq {
    //we want to pair title and abstract, title and abstract pair together:
    let mutable title = ""
    let mutable abstractxt = "" 
    while xml.Read() do
        match xml.NodeType with
        | System.Xml.XmlNodeType.Element ->
            if xml.Name = "title" then
                title <- xml.ReadElementContentAsString()
            elif xml.Name = "abstract" then
                abstractxt <- xml.ReadElementContentAsString()
        | System.Xml.XmlNodeType.EndElement ->
            //if abstract text starts with "|" then skip. Len must be > 4 words
            if xml.Name = "doc"
                && not (abstractxt.StartsWith("|"))
                && abstractxt.Split(' ').Length > 4
                //does not (contain | and thumb) or (thumb and px)
                && (not (
                        abstractxt.Contains("|")
                        && abstractxt.Contains("thumb")
                    )
                    || not (
                        abstractxt.Contains("thumb")
                        && abstractxt.Contains("px")
                    )) then

                //strip "Wikipedia:" from title
                yield {Title = title.Replace("Wikipedia: ", ""); Text = abstractxt}
        | _ -> ()
    }
//Question: What is the function loop doing?
//Answer: It is a recursive function that returns a sequence of documents. The sequence is created by looping through the xml file and extracting the title and abstract text from the xml file.  
//Question: What name better reflects what the function does?
//Answer: extractDocsFromXml
//Question: What is the type of the function?
//Answer: System.Xml.XmlReader -> seq<Doc>
//Question: The type is actually DocItem. What should we rename the function to incorporate this?
//Answer: extractDocItemsFromXml
 
 
xml.Close()
xml.Dispose()
let all = loop xml |> Seq.toArray

sv.Process all

//how many titles start with "A"
all|> Array.filter (fun (t, _) -> t.Contains "Machine l") //|> Array.length

138782 + all.Length

//close explorer
System.Diagnostics.Process.GetProcessesByName("explorer") |> Array.iter (fun p -> p.Kill())

