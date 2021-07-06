#I @"C:\Users\cybernetic\source\repos\" 
#r @"Prelude\Prelude\bin\Release\net5\Prelude.dll"
#load "tensor-extensions.fsx"

open Microsoft.ML.OnnxRuntime
open System
open Prelude.Common 
open ``Tensor-extensions``
open Prelude

//Do max sentences instead of maxlen
let decoderSampler (decoder:InferenceSession) sampler stopToken maxlen encoderStates =
    
    let generatedTokens = ResizeArray<int>([|0|])
    let mutable generated = Tensors.ArrayTensorExtensions.ToTensor(array2D [ [ 0 ] ])

    let mutable c = 0
    let mutable stop = false
    let sw = Diagnostics.Stopwatch()

    while c < maxlen && not stop do 
        let logits =
            decoder.Run 
                [| NamedOnnxValue.CreateFromTensor("input_ids", generated)
                   encoderStates |] 

        let decoded = Seq.head logits

        let result = decoded.AsTensor<float32>()   
         
        let index = sampler result.[0, ^0, *] 

        if index <> stopToken then generatedTokens.Add index  
        
        generated <- 
            Tensors.ArrayTensorExtensions.ToTensor(array2D [| generatedTokens.ToArray() |])
             
        decoded.Dispose()

        logits.Dispose()

        sw.Stop() 
        c <- c + 1
        stop <- index = stopToken

    generatedTokens.ToArray()

let basicSampler k p (choices:float32[]) =
    let probs = [for i in 0..choices.Length - 1 -> i, exp (float choices.[i])] |> List.normalizeWeights
    let choices = SampleSummarize.getLargeProbItems p probs |> List.toArray            
    if k > 0 then
        choices 
        |> Array.rev 
        |> Array.take k
        |> Sampling.discreteSample
    else Sampling.discreteSample choices
 
 

let nnSampler (encoder: InferenceSession) (decoder: InferenceSession) detokenizer stoptoken sampler 
    seqlen toks =
    [| for tok in toks do
        let t =
            Tensors.ArrayTensorExtensions.ToTensor(array2D [| tok |])

        let outputs =
            encoder.Run [| NamedOnnxValue.CreateFromTensor("input_ids", t) |]

        let encoderStates = Seq.head outputs

        encoderStates.Name <- "encoder_hidden_states"

        let str =
            decoderSampler decoder sampler stoptoken seqlen encoderStates

        encoderStates.Dispose()
        outputs.Dispose()
        yield detokenizer str |] 