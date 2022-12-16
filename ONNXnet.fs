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
    type NNet<'a>(?loc: string, ?InferenceSessionOptions, ?inferenceSession, ?InputIndex, ?AttentionIndex) =
        let disposables = ResizeArray()
        let sessionOptions = InferenceSessionOptions

        let attentionIndex = defaultArg AttentionIndex 1
        let inputIndex = defaultArg InputIndex 0

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
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[inputIndex], Tensors.ArrayTensorExtensions.ToTensor(input)) |]

            disposables.Add result
            Seq.toArray result

        member __.RunT(input: _ [,,,]) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[inputIndex], Tensors.ArrayTensorExtensions.ToTensor(input)) |]

            disposables.Add result
            disposablesToTensorArray result 

        member __.Run(input: Tensors.DenseTensor<'a>) =
            let result = internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[inputIndex], input) |]
            disposables.Add result
            Seq.toArray result

        member __.Run(input: Tokenized<int>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[inputIndex], input.TokenIds)
                                   NamedOnnxValue.CreateFromTensor(keys[attentionIndex], input.AttentionMasks) |] 

            disposables.Add result
            Seq.toArray result

        member nn.RunT(input: Tokenized<int>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[inputIndex], input.TokenIds)
                                   NamedOnnxValue.CreateFromTensor(keys[attentionIndex], input.AttentionMasks) |]

            disposables.Add result
            disposablesToTensorArray result 

        member nn.RunT(input: Tokenized<int64>) =
            let result =
                internalNet.Run [| NamedOnnxValue.CreateFromTensor(keys[inputIndex], input.TokenIds)
                                   NamedOnnxValue.CreateFromTensor(keys[attentionIndex], input.AttentionMasks) |]

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
        
        member __.InputIndex = inputIndex

        member __.AttentionIndex = attentionIndex

        member __.InputMetaData = internalNet.InputMetadata

        member __.OutputMetaData = internalNet.OutputMetadata

        member __.InternalSession = internalNet

        interface IDisposable with
            member t.Dispose() = t.Dispose()
  
      
    type DecoderSampler(decoder:NNet<float32>, decodingStartToken, stopToken, ?blockStartTokenIndex, ?sentenceEndTokens : int[]) = 
                
        let blockStartTokenIndex = defaultArg blockStartTokenIndex 3
        
        let sentenceEndTokens = HashSet(defaultArg sentenceEndTokens [||])
          
        let repeatmem = ResizeArray()  

        let tokenEvent = Event<_>()
        
        let samplerinfo = {
            DecodingStartToken = decodingStartToken
            StopToken = stopToken
            BlockStartTokenIndex = blockStartTokenIndex
            MinSentLen = 0
        }      
        
        //Do max sentences instead of maxlen
        let decoderSampler samplerfn decoderInputName maxlen minlen countsents (encoderStates : _ option) (initialTokens : _ [])= 
        
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
                //minsentlen record, minlen option
                let samplerinfo' = match minlen with Some l -> {samplerinfo with MinSentLen = l} | _ -> samplerinfo
                let index, p = samplerfn samplerinfo' {SampleLen = c} logits[0].[0, ^0, *]  
                
                repeatmem.Add index

                if repeatmem.Count > 5 then repeatmem.RemoveAt 0  

                if index <> stopToken then 
                    generatedTokens.Add index  
                    tokenEvent.Trigger(index)
                    probs.Add p
                
                generated <- Tensor.ofNestedSeq2D [| generatedTokens.ToArray() |]
                    
                decoder.GC()
                 
                if countsents && sentenceEndTokens.Count > 0 then 
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

        member __.TokenEvent = tokenEvent.Publish 
 
        member t.RunDecoderStep (input: _ [], ?EncoderState) =
            decoderStep
                decoder.InputKeys[decoder.InputIndex]
                EncoderState
                input

        member t.RunDecoderSampler(?minlen, ?seqlen, ?countBySentences, ?sampler, ?input, ?encoderHiddenState) =  
            let samplerfn = defaultArg sampler Sampling.argmax 
            
            decoderSampler
                samplerfn
                decoder.InputKeys[decoder.InputIndex]
                (defaultArg seqlen 120)
                minlen
                (defaultArg countBySentences false)
                encoderHiddenState 
                (defaultArg input [| decodingStartToken |])  

        member t.SamplerInfo = samplerinfo  
 
        member __.SentenceEndTokens = sentenceEndTokens
        
 
        interface IDisposable with 
            member t.Dispose() = 
                decoder.Dispose()
                repeatmem.Clear()     
        
    type EncoderDecoderSampler(encoder:NNet<float32>, decoder:NNet<float32>, decodingStartToken, stopToken, ?blockStartTokenAtIndex, ?sentenceEndTokens : int []) = 
        
        let decoderModel = new DecoderSampler(decoder, decodingStartToken, stopToken, ?sentenceEndTokens = sentenceEndTokens, ?blockStartTokenIndex = blockStartTokenAtIndex)
        
        let mutable currentEncoderState = None 

        member private m.encoderDecoderSampler4D encHiddenStatesName seqlen (encoderInput: 'a [,,,]) =  
            encoder.GC()

            let h = encoder.Run(encoderInput)[0]

            h.Name <- encHiddenStatesName 
            
            let tokens = decoderModel.RunDecoderSampler(seqlen = seqlen, encoderHiddenState = h) 

            encoder.GC()
            tokens 

        member private m.encoderDecoderSampler 
            encHiddenStatesName
            minlen
            seqlen
            countBySentences
            sampler
            (encoderInput: _ [])
            (decoderInput: _ []) =        
            encoder.GC()
            let attentionMask = Array.create encoderInput.Length 1 
            let keys = (m.Encoder : NNet<float32>).InputKeys 
            let input =
                [| [ encoderInput ]
                   |> Tensor.ofNestedSeq2D
                   |> Tensor.toOnnxValue keys[m.Encoder.InputIndex]
                   [ attentionMask ]
                   |> Tensor.ofNestedSeq2D
                   |> Tensor.toOnnxValue keys[m.Encoder.AttentionIndex] |]

            let h = encoder.Run(input)[0]

            h.Name <- encHiddenStatesName

            let tokens =
                decoderModel.RunDecoderSampler( 
                    ?minlen = minlen,
                    seqlen = seqlen,
                    ?countBySentences = countBySentences,
                    ?sampler = sampler,
                    encoderHiddenState = h,
                    input = decoderInput
                )

            encoder.GC()
            tokens

        member __.Encoder = encoder : NNet<float32>

        member __.Decoder = decoder : NNet<float32> 
        
        member __.SentenceEndTokens = sentenceEndTokens

        member t.Run(input: _ [,,,], ?seqlen, ?EncoderHiddenStatesName) = 
            t.encoderDecoderSampler4D
                (defaultArg EncoderHiddenStatesName "encoder_hidden_states") 
                (defaultArg seqlen 120)
                input 

        member t.Run(input: int [], ?decoderInput, ?minlen, ?seqlen, ?countBySentences, ?sampler, ?EncoderHiddenStatesName) = 
            t.encoderDecoderSampler
                (defaultArg EncoderHiddenStatesName "encoder_hidden_states") 
                minlen
                (defaultArg seqlen 120)
                countBySentences 
                sampler
                input
                (defaultArg decoderInput [| decodingStartToken |])

        member t.RunDecoderStepLogits (input: _ [], ?encoderHiddenState, ?DecoderInputName) = 
            let encoderstate = 
                match encoderHiddenState with
                | Some h -> h 
                | None ->  
                    match currentEncoderState with 
                    | None -> failwith "No encoder state"
                    | Some h -> h 

            decoderModel.RunDecoderStep(input, EncoderState = encoderstate)
              
        member t.RunDecoderStep(input: _ [], ?encoderHiddenState, ?DecoderInputName) =
            //take logits then convert into a probability distribution 
            let logits = t.RunDecoderStepLogits(input, ?encoderHiddenState = encoderHiddenState, ?DecoderInputName = DecoderInputName)
            [|for i in 0 .. logits.Length - 1 -> i, exp(logits.[i])|]
            |> Array.normalize

        member __.EncoderHiddenState with get() = currentEncoderState and set h = currentEncoderState <- h

        member __.TokenEvent = decoderModel.TokenEvent
        
        member t.RunDecoderSampler(?input, ?seqlen, ?CountBySentences, ?encoderHiddenState) = 
            let encoderstate = 
                match encoderHiddenState with
                | Some h -> h 
                | None ->  
                    match currentEncoderState with 
                    | None -> failwith "No encoder state"
                    | Some h -> h 
            
            decoderModel.RunDecoderSampler(?input = input, ?seqlen = seqlen, ?countBySentences = CountBySentences, encoderHiddenState = encoderstate)

                
        member t.RunEncoder(input: Tokenized<int>, ?encoderHiddenStatesName) = 
            let input =  [| NamedOnnxValue.CreateFromTensor(encoder.InputKeys[encoder.InputIndex], input.TokenIds);
                            NamedOnnxValue.CreateFromTensor(encoder.InputKeys[encoder.AttentionIndex], input.AttentionMasks) |]
            
            encoder.GC()
            let h = encoder.Run(input)[0]
            h.Name <- defaultArg encoderHiddenStatesName "encoder_hidden_states"
            currentEncoderState <- Some h
           
        member t.EncoderState = currentEncoderState

        interface IDisposable with 
            member t.Dispose() = 
                encoder.Dispose()
                decoder.Dispose()
                currentEncoderState |> Option.iter (fun t -> t.Dispose())
                currentEncoderState <- None
                

    
                