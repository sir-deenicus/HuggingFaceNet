namespace TensorNet

open System.Collections.Generic
open Microsoft.ML.OnnxRuntime
open System
open TensorExtensions

[<RequireQualifiedAccess>]
module ONNX =
    type AccleratorProvider =
    | CPU
    | CUDA

    type InferenceSessionOptions(GraphOptimizationLevel, ?AcceleratorProvider, ?GraphSerlializationPath) =
        let sessionOptions = 
            match AcceleratorProvider with
            | Some CUDA -> SessionOptions.MakeSessionOptionWithCudaProvider (GraphOptimizationLevel = GraphOptimizationLevel)
            | Some CPU
            | None -> new SessionOptions(GraphOptimizationLevel = GraphOptimizationLevel)
        do 
            match GraphSerlializationPath with
            | Some path -> sessionOptions.OptimizedModelFilePath <- path
            | None -> () 

        member __.Options = sessionOptions

        member __.Dispose() =
            sessionOptions.Dispose()

        interface IDisposable with
            member s.Dispose() = s.Dispose()
        
    //meets disposable interface
    type NNet<'a>(?loc: string, ?InferenceSessionOptions, ?inferenceSession) =
        let disposables = ResizeArray()
        let sessionOptions = InferenceSessionOptions

        let newInternalNet() =
            match (inferenceSession, loc) with
            | Some session, _ -> session
            | None, Some loc ->
                match InferenceSessionOptions with
                | Some (sessionOptions: InferenceSessionOptions) -> new InferenceSession(loc, sessionOptions.Options)
                | None -> new InferenceSession(loc)
            | None, None -> failwith "Either inferenceSession or loc must be specified"

        let mutable internalNet = newInternalNet()
        let keys = Seq.toArray internalNet.InputMetadata.Keys

        member __.Run(input: _ [,,,]) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], Tensors.ArrayTensorExtensions.ToTensor(input)) |]

            disposables.Add result
            Seq.toArray result

        member __.RunT(input: _ [,,,]) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], Tensors.ArrayTensorExtensions.ToTensor(input)) |]

            disposables.Add result
            [| for r in result -> r.AsTensor<'a>() |]

        member __.Run(input: Tensors.DenseTensor<'a>) =
            let result = internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], input) |]
            disposables.Add result
            Seq.toArray result

        member __.Run(input: Tensors.DenseTensor<int>, attention_mask: Tensors.DenseTensor<int>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], input)
                                   NamedOnnxValue.CreateFromTensor(keys[1], attention_mask) |]

            disposables.Add result
            Seq.toArray result

        member __.RunT(input: Tensors.DenseTensor<int>, attention_mask: Tensors.DenseTensor<int>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], input)
                                   NamedOnnxValue.CreateFromTensor(keys[1], attention_mask) |]

            disposables.Add result
            [| for r in result -> r.AsTensor<'a>() |]

        member __.RunT(input: Tensors.DenseTensor<int64>, attention_mask: Tensors.DenseTensor<int64>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], input)
                                   NamedOnnxValue.CreateFromTensor(keys[1], attention_mask) |]

            disposables.Add result
            [| for r in result -> r.AsTensor<'a>() |]

        member __.Run(input: Collections.Generic.IDictionary<string, Tensors.DenseTensor<'a>>) =
            let result =
                internalNet.Run [| for KeyValue (key, tensor) in input -> NamedOnnxValue.CreateFromTensor(key, tensor) |]

            disposables.Add result
            Seq.toArray result

        member __.RunT(input: seq<string * Tensors.Tensor<'a>>) =
            let result =
                internalNet.Run [| for (key, tensor) in input -> NamedOnnxValue.CreateFromTensor(key, tensor) |]

            disposables.Add result
            [| for r in result -> r.AsTensor<'a>() |]

        member __.Run(input: (string * Tensors.DenseTensor<'a>) []) =
            let result =
                internalNet.Run [| for key, tensor in input -> NamedOnnxValue.CreateFromTensor(key, tensor) |]

            disposables.Add result
            Seq.toArray result

        member __.Run(input: NamedOnnxValue []) =
            let result = internalNet.Run input
            disposables.Add result
            Seq.toArray result

        member __.RunT(input: NamedOnnxValue []) =
            let result = internalNet.Run input
            disposables.Add result
            [| for r in result -> r.AsTensor<'a>() |]

        member __.GC() =
            for d in disposables do
                for t in d do
                    t.Dispose()

                d.Dispose()

            disposables.Clear() 
            
        member t.Dispose() =
            t.GC()
            internalNet.Dispose()

            match sessionOptions with
            | None -> ()
            | Some sessionOptions -> sessionOptions.Dispose()
            
        member t.ResetMemory() =
            t.GC()
            internalNet.Dispose()
            internalNet <- newInternalNet() 

        member __.InputKeys = keys

        member __.InputMetaData = internalNet.InputMetadata

        member __.OutputMetaData = internalNet.OutputMetadata

        member __.InternalSession = internalNet

        interface IDisposable with
            member t.Dispose() = t.Dispose()
            

    type EncoderDecoderSampler(encoder:NNet<float32>, decoder:NNet<float32>, startToken, stopToken, ?sampler, ?batchsize, ?blockStartTokenAtIndex, ?sentenceEndTokens : HashSet<int64>, ?countBySentences) = 
        let sentenceEndTokens = defaultArg sentenceEndTokens (HashSet())
        let countsents = defaultArg countBySentences false

        //fail countsents is true and sentendtoks is empty
        do if countsents && sentenceEndTokens.Count = 0 then failwith "countBySentences is true but sentenceEndTokens is empty" 
        let batchsize = defaultArg batchsize 1
        let blockStartTokenAtIndex = defaultArg blockStartTokenAtIndex 3

        let mutable currentEncoderState = None

        let mem = ResizeArray() 
        let blocked = HashSet()
        let argmax (a:_[]) =
            if mem.Count >= blockStartTokenAtIndex then blocked.Add (startToken) |> ignore
            Array.argmaxf32b blocked a

        let samplerfn = defaultArg sampler argmax  

        //Do max sentences instead of maxlen
        let decoderSampler decoderInputName maxlen encoderStates = 
        
            let generatedTokens = ResizeArray<int>([| startToken |])
            let probs = ResizeArray()
            let mutable generated = Tensor.ofNestedSeq2D [ [ startToken  ] ]

            let mutable c = 0
            let mutable stop = false 
            mem.Clear()

            while c < maxlen && not stop do 
                let logits =
                    decoder.RunT(
                        [| generated.ToOnnxValue decoderInputName
                           encoderStates |]
                    ) 

                let index, p = samplerfn logits[0].[0, ^0, *] 

                mem.Add index

                if mem.Count > 5 then mem.RemoveAt 0  

                if index <> stopToken then 
                    generatedTokens.Add index  
                    probs.Add p
                
                generated <- Tensor.ofNestedSeq2D [| generatedTokens.ToArray() |]
                    
                decoder.GC()
                
                if countsents then 
                    if sentenceEndTokens.Contains index then c <- c + 1
                else c <- c + 1
                
                if mem.Count >= 5 && HashSet(mem).Count = 1 then stop <- true
                else stop <- int64 index = stopToken

            blocked.Clear()
            generatedTokens.ToArray(), probs.ToArray()

        let decoderStep decoderInputName encoderStates (tokens:_[]) =
            let generated = Tensor.ofNestedSeq2D [ tokens ]
            let logits =
                decoder.RunT(
                    [| generated.ToOnnxValue decoderInputName
                       encoderStates |]
                )

            logits[0].[0, ^0, *]


        member private __.encoderDecoderSampler4D encHiddenStatesName decoderInputsName seqlen (encoderInput: 'a [,,,]) =  
            encoder.GC()

            let h = encoder.Run(encoderInput)[0]

            h.Name <- encHiddenStatesName 

            let str, ps =
                decoderSampler decoderInputsName seqlen h

            encoder.GC()
            str, ps  

        member private __.encoderDecoderSampler inputname encHiddenStatesName attentionMaskName decoderInputsName seqlen (encoderInput: _ []) =  
            encoder.GC()
            let attentionMask = Array.create encoderInput.Length 1 
            let input  = [| [ encoderInput ] |> Tensor.ofNestedSeq2D |> Tensor.toOnnxValue inputname
                            [ attentionMask] |> Tensor.ofNestedSeq2D |> Tensor.toOnnxValue attentionMaskName |]
            
            let h = encoder.Run(input)[0]

            h.Name <- encHiddenStatesName 

            let str, ps =
                decoderSampler decoderInputsName seqlen h

            encoder.GC()
            str, ps  

        member __.Encoder = encoder

        member __.Decoder = decoder

        member t.Run(input: _ [,,,], ?seqlen, ?EncoderHiddenStatesName, ?DecoderInputsName) = 
            t.encoderDecoderSampler4D
                (defaultArg EncoderHiddenStatesName "encoder_hidden_states") 
                (defaultArg DecoderInputsName "input_ids")
                (defaultArg seqlen 120)
                input 

        member t.Run(input: int [], ?seqlen, ?InputName, ?EncoderHiddenStatesName, ?DecoderInputsName, ?AttentionMaskName) = 
            t.encoderDecoderSampler
                (defaultArg InputName "input_ids")
                (defaultArg EncoderHiddenStatesName "encoder_hidden_states")
                (defaultArg AttentionMaskName "attention_mask")
                (defaultArg DecoderInputsName "input_ids") 
                (defaultArg seqlen 120)
                input

        member t.RunDecoderStep (input: _ [], ?encoderHiddenState, ?DecoderInputsName) = 
            let encoderstate = 
                match encoderHiddenState with
                | Some h -> h 
                | None ->  
                    match currentEncoderState with 
                    | None -> failwith "No encoder state"
                    | Some h -> h 

            decoderStep
                (defaultArg DecoderInputsName "input_ids")
                encoderstate 
                input

        member __.EncoderHiddenState with get() = currentEncoderState and set h = currentEncoderState <- h
        member t.RunDecoderSampler(?encoderHiddenState, ?seqlen, ?DecoderInputsName) = 
            let encoderstate = 
                match encoderHiddenState with
                | Some h -> h 
                | None ->  
                    match currentEncoderState with 
                    | None -> failwith "No encoder state"
                    | Some h -> h 
        
            decoderSampler
                (defaultArg DecoderInputsName "input_ids")
                (defaultArg seqlen 120)
                encoderstate  

                
        member t.RunEncoder(input : _ []) =
            encoder.GC()
            let input  = [| [ input ] |> array2D |> Array2D.toTensor |> Tensor.toOnnxValue "input_ids" |]
            let h = encoder.Run(input)[0]
            currentEncoderState <- Some h
            h

        interface IDisposable with 
            member t.Dispose() = 
                encoder.Dispose()
                decoder.Dispose()
                currentEncoderState |> Option.iter (fun t -> t.Dispose())
                currentEncoderState <- None
                mem.Clear()
                blocked.Clear() 

    type DecoderSampler(startToken, stopToken, decoder:NNet<float32>, ?sampler, ?batchsize, ?blockStartTokenAtIndex, ?sentenceEndTokens : HashSet<int64>, ?countBySentences) = 
        let sentenceEndTokens = defaultArg sentenceEndTokens (HashSet())
        let countsents = defaultArg countBySentences false

        //fail countsents is true and sentendtoks is empty
        do if countsents && sentenceEndTokens.Count = 0 then failwith "countBySentences is true but sentenceEndTokens is empty" 
        let batchsize = defaultArg batchsize 1
        let blockStartTokenAtIndex = defaultArg blockStartTokenAtIndex 3

        let mem = ResizeArray() 
        let blocked = HashSet()
        let argmax (a:_[]) =
            if mem.Count >= blockStartTokenAtIndex then blocked.Add (int startToken) |> ignore
            Array.argmaxf32b blocked a

        let samplerfn = defaultArg sampler argmax  

        //Do max sentences instead of maxlen
        let decoderSampler decoderInputName maxlen = 
        
            let generatedTokens = ResizeArray<int64>([|startToken|])
            let probs = ResizeArray()
            let mutable generated = Tensor.ofNestedSeq2D [ [ startToken ] ]

            let mutable c = 0
            let mutable stop = false 
            mem.Clear()

            while c < maxlen && not stop do 
                let logits =
                    decoder.RunT(
                        [| generated.ToOnnxValue decoderInputName  |]
                    ) 

                let index, p = samplerfn logits[0].[0, ^0, *] 

                mem.Add index

                if mem.Count > 5 then mem.RemoveAt 0  

                if index <> stopToken then 
                    generatedTokens.Add index  
                    probs.Add p
                
                generated <- Tensor.ofNestedSeq2D [| generatedTokens.ToArray() |]
                    
                decoder.GC()
                
                if countsents then 
                    if sentenceEndTokens.Contains index then c <- c + 1
                    
                else c <- c + 1
                
                if mem.Count >= 5 && HashSet(mem).Count = 1 then stop <- true
                else stop <- int64 index = stopToken

            blocked.Clear()
            generatedTokens.ToArray(), probs.ToArray()

        let decoderStep decoderInputName (tokens:int64[]) =
            let generated = Tensor.ofNestedSeq2D [ tokens ]
            let logits =
                decoder.RunT(
                    [| generated.ToOnnxValue decoderInputName|]
                )

            logits[0].[0, ^0, *] 
 

        member __.Decoder = decoder
 
        member t.RunDecoderStep (input: _ [],  ?DecoderInputsName) =
            decoderStep
                (defaultArg DecoderInputsName "input_ids")
                input
 
        interface IDisposable with 
            member t.Dispose() = 
                decoder.Dispose()
                mem.Clear()
                blocked.Clear()
                