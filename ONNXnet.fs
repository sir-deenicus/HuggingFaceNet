namespace TensorNet

open System.Collections.Generic
open Microsoft.ML.OnnxRuntime
open System
open TensorNet.Tokenizers
open TensorExtensions 
open Sampling
        
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
        
        let disposablesToTensorArray (disposableCollection:IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) =
            [| for r in disposableCollection -> r.AsTensor<'a>() |]
            
        member __.Run(input: _ [,,,]) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], Tensors.ArrayTensorExtensions.ToTensor(input)) |]

            disposables.Add result
            Seq.toArray result

        member __.RunT(input: _ [,,,]) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], Tensors.ArrayTensorExtensions.ToTensor(input)) |]

            disposables.Add result
            disposablesToTensorArray result 

        member __.Run(input: Tensors.DenseTensor<'a>) =
            let result = internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], input) |]
            disposables.Add result
            Seq.toArray result

        member __.Run(input: Tokenized<int>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], input.TokenIds)
                                   NamedOnnxValue.CreateFromTensor(keys[1], input.AttentionMasks) |]

            disposables.Add result
            Seq.toArray result

        member nn.RunT(input: Tokenized<int>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], input.TokenIds)
                                   NamedOnnxValue.CreateFromTensor(keys[1], input.AttentionMasks) |]

            disposables.Add result
            disposablesToTensorArray result 

        member nn.RunT(input: Tokenized<int64>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[0], input.TokenIds)
                                   NamedOnnxValue.CreateFromTensor(keys[1], input.AttentionMasks) |]

            disposables.Add result
            [| for r in result -> r.AsTensor<'a>() |] 

        member __.RunT(input: seq<string * Tensors.DenseTensor<'a>>) =
            let result =
                internalNet.Run [| for (key, tensor) in input -> NamedOnnxValue.CreateFromTensor(key, tensor) |]

            disposables.Add result
            disposablesToTensorArray result 
              
        member __.Run(input: NamedOnnxValue []) =
            let result = internalNet.Run input
            disposables.Add result
            Seq.toArray result

        member __.RunT(input: NamedOnnxValue []) =
            let result = internalNet.Run input
            disposables.Add result
            disposablesToTensorArray result 

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
  
      
    type DecoderSampler(startToken, stopToken, decoder:NNet<float32>, ?sampler, ?minSentenceLength, ?batchsize, ?blockStartTokenIndex, ?sentenceEndTokens : int[], ?countBySentences) = 
        
        let countsents = defaultArg countBySentences false 
        
        let minsentlen = defaultArg minSentenceLength 0
        
        let batchsize = defaultArg batchsize 1
        
        let blockStartTokenIndex = defaultArg blockStartTokenIndex 3
        
        let sentenceEndTokens = HashSet(defaultArg sentenceEndTokens [||])
         
        //fail countsents is true and sentendtoks is empty
        do if countsents && sentenceEndTokens.Count = 0 then failwith "countBySentences is true but sentenceEndTokens is empty" 
         
        let repeatmem = ResizeArray()  
        
        let samplerinfo = {
            StartToken = startToken
            StopToken = stopToken
            BlockStartTokenIndex = blockStartTokenIndex
            MinSentLen = minsentlen 
        }     
            
        let samplerfn = defaultArg sampler Sampling.argmax  
        
        //Do max sentences instead of maxlen
        let decoderSampler decoderInputName maxlen (encoderStates : _ option) (initialTokens : _ [])= 
        
            let generatedTokens = ResizeArray<int>(initialTokens)
            let probs = ResizeArray()
            let mutable generated = Tensor.ofNestedSeq2D [ initialTokens ]

            let mutable c = 0
            let mutable stop = false 
            repeatmem.Clear()

            while c < maxlen && not stop do 
                let logits =
                    decoder.RunT(
                        [| yield generated.ToOnnxValue decoderInputName
                           if encoderStates.IsSome then yield encoderStates.Value |]
                    ) 

                let index, p = samplerfn samplerinfo {SampleLen = c} logits[0].[0, ^0, *] 
                
                repeatmem.Add index

                if repeatmem.Count > 5 then repeatmem.RemoveAt 0  

                if index <> stopToken then 
                    generatedTokens.Add index  
                    probs.Add p
                
                generated <- Tensor.ofNestedSeq2D [| generatedTokens.ToArray() |]
                    
                decoder.GC()
                
                if countsents then 
                    if sentenceEndTokens.Contains index then c <- c + 1
                else c <- c + 1
                
                if repeatmem.Count >= 5 && HashSet(repeatmem).Count = 1 then stop <- true
                else stop <- index = stopToken

            {Tokens = generatedTokens.ToArray(); Probabilities = probs.ToArray()}

        let decoderStep decoderInputName (encoderStates : _ option) (tokens:_[]) =
            let generated = Tensor.ofNestedSeq2D [ tokens ]
            let logits =
                decoder.RunT(
                    [| yield generated.ToOnnxValue decoderInputName
                       if encoderStates.IsSome then yield encoderStates.Value |]
                )

            logits[0].[0, ^0, *]

        member __.Decoder = decoder
 
        member t.RunDecoderStep (input: _ [],  ?DecoderInputName, ?EncoderState) =
            decoderStep
                (defaultArg DecoderInputName "input_ids")
                EncoderState
                input

        member t.RunDecoderSampler(?encoderHiddenState, ?seqlen, ?DecoderInputsName, ?input) =  
            decoderSampler
                (defaultArg DecoderInputsName "input_ids")
                (defaultArg seqlen 120)
                encoderHiddenState 
                (defaultArg input [|startToken|])
 
        interface IDisposable with 
            member t.Dispose() = 
                decoder.Dispose()
                repeatmem.Clear() 
                
                
    type EncoderDecoderSampler(encoder:NNet<float32>, decoder:NNet<float32>, startToken, stopToken, ?sampler, ?batchsize, ?blockStartTokenAtIndex, ?sentenceEndTokens : int [], ?minsentlen, ?countBySentences) = 
        
        let decoderModel = new DecoderSampler(startToken, stopToken, decoder)
        
        let mutable currentEncoderState = None 

        member private __.encoderDecoderSampler4D encHiddenStatesName decoderInputsName seqlen (encoderInput: 'a [,,,]) =  
            encoder.GC()

            let h = encoder.Run(encoderInput)[0]

            h.Name <- encHiddenStatesName 

            let tokens = decoderModel.RunDecoderSampler(h, seqlen, decoderInputsName) 

            encoder.GC()
            tokens 

        member private __.encoderDecoderSampler inputname encHiddenStatesName attentionMaskName decoderInputsName seqlen (encoderInput: _ []) =  
            encoder.GC()
            let attentionMask = Array.create encoderInput.Length 1 
            let input  = [| [ encoderInput ] |> Tensor.ofNestedSeq2D |> Tensor.toOnnxValue inputname
                            [ attentionMask] |> Tensor.ofNestedSeq2D |> Tensor.toOnnxValue attentionMaskName |]
            
            let h = encoder.Run(input)[0]

            h.Name <- encHiddenStatesName 

            let tokens = decoderModel.RunDecoderSampler(h, seqlen, decoderInputsName) 

            encoder.GC()
            tokens

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

        member t.RunDecoderStep (input: _ [], ?encoderHiddenState, ?DecoderInputName) = 
            let encoderstate = 
                match encoderHiddenState with
                | Some h -> h 
                | None ->  
                    match currentEncoderState with 
                    | None -> failwith "No encoder state"
                    | Some h -> h 

            decoderModel.RunDecoderStep(input, ?DecoderInputName = DecoderInputName, EncoderState = encoderstate)
              

        member __.EncoderHiddenState with get() = currentEncoderState and set h = currentEncoderState <- h
        
        member t.RunDecoderSampler(?encoderHiddenState, ?seqlen, ?DecoderInputsName) = 
            let encoderstate = 
                match encoderHiddenState with
                | Some h -> h 
                | None ->  
                    match currentEncoderState with 
                    | None -> failwith "No encoder state"
                    | Some h -> h 
            
            decoderModel.RunDecoderSampler(encoderstate, ?seqlen = seqlen, ?DecoderInputsName = DecoderInputsName)

                
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
                

    
                