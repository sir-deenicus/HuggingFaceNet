namespace TensorNet

open System.Collections.Generic
open Microsoft.ML.OnnxRuntime
open System
open TensorNet.Tokenizers
open TensorExtensions
   
type SamplerOutput = {
    Tokens : int []
    Probabilities : float32 [] 
} 

type SamplerInfo = {
    DecodingStartToken : int option
    StopToken : int 
    BlockStartTokenIndex : int 
    MinSentLen : int
}

type SamplerState = {
    SampleLen : int 
}

module Array = 
    let takeOrMax n (a:_[]) = a.[..(min (a.Length - 1) n)]
    let inline normalize (l: (_ * float32) []) =
        let tot = Array.sumBy snd l
        Array.Parallel.map (fun (x, w) -> x, w / tot) l

module Sampling =
    let random = System.Random()

    let getTopProbs maxp (ps: _ []) = 
        let top = ResizeArray(ps.Length)
        let mutable cumulativeprob = 0.f
        let mutable k = 0
        let mutable notdone = true

        while k < ps.Length && notdone do
            let i, p = ps.[k]
            cumulativeprob <- cumulativeprob + p
            top.Add(i, p)

            if cumulativeprob > maxp then
                notdone <- false

            k <- k + 1

        top.ToArray()

    let getLargeProbItems maxp (ps: _ []) =
        let ps' = Array.sortByDescending snd ps
        getTopProbs maxp ps'

    let discreteSampleIndex (prob: _ []) =
        let cummulativeDistr = Array.create prob.Length (snd prob.[0])

        for i in 1 .. prob.Length - 1 do
            cummulativeDistr.[i] <- cummulativeDistr.[i - 1] + (snd prob.[i])

        let k =
            float32 (random.NextDouble())
            * cummulativeDistr.[^0]

        let rec cummProb index =
            if k > cummulativeDistr.[index] then
                cummProb (index + 1)
            else
                index

        cummProb 0

    let argmax samplingInfo samplerState (logits: _ []) = 
        if samplerState.SampleLen >= samplingInfo.BlockStartTokenIndex && samplingInfo.DecodingStartToken.IsSome then
            logits.[samplingInfo.DecodingStartToken.Value] <- -infinityf 

        //suppress end of sentence tokens while < minsentlen
        if samplerState.SampleLen < samplingInfo.MinSentLen then
            logits.[samplingInfo.StopToken] <- -infinityf

        Array.argmaxf32 logits


    let sample k p samplingInfo samplerState (logits: float32 []) =  
        if samplerState.SampleLen >= samplingInfo.BlockStartTokenIndex && samplingInfo.DecodingStartToken.IsSome then
            logits.[samplingInfo.DecodingStartToken.Value] <- -infinityf

        //what is the above line of code doing?
        //Answer: it sets the probability of the start token to -infinity, so that it will never be chosen as the next token
        //on the condition that the sample length is greater than or equal to the block start token index
        //The reason for this is that the start token is used to indicate the start of a sentence, and we don't want to start a sentence in the middle of a sentence

        //suppress end of sentence tokens while < minsentlen
        if samplerState.SampleLen < samplingInfo.MinSentLen then
            logits.[samplingInfo.StopToken] <- -infinityf

        let probs =
            [| for i in 0 .. logits.Length - 1 -> i, exp (float32 logits.[i]) |]
            |> Array.normalize

        let choices = getLargeProbItems p probs

        let i =
            if k > 0 then
                choices |> Array.takeOrMax k |> discreteSampleIndex
            else
                discreteSampleIndex choices

        choices[i]

    let sampleTypical k p samplingInfo samplerState (logits: float32 []) =  
        if samplerState.SampleLen >= samplingInfo.BlockStartTokenIndex && samplingInfo.DecodingStartToken.IsSome then
            logits.[samplingInfo.DecodingStartToken.Value] <- -infinityf

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
        //what does the above line of code do?
        //Answer: it sorts the probabilities by the absolute value of the difference between the log of the probability and the entropy of the probabilities (i.e. it sorts the probabilities by how different they are from the average probability)
     
        let choices = 
            if k = 0 then getTopProbs p sorted 
            else sorted |> Array.takeOrMax k |> getTopProbs p

        let i = discreteSampleIndex choices

        choices[i]

        
    let subsetNucleus k p (probs: (_ * float32) []) = 
        let choices = getLargeProbItems p probs   
        if k > 0 then
            choices |> Array.takeOrMax k  
        else choices 

    
    let subsetTypical k p (probs: (_ * float32) []) =   
        //what does the above line of code do?
        //Answer: it normalizes the probabilities so that they sum to 1

        let ent = -1f * (Array.sumBy (fun (_, p) -> p * log (p + 1e-30f)) probs)
        //what does the above line of code do?
        //Answer: it calculates the entropy of the probabilities
        
        let sorted = probs |> Array.sortBy (fun (_, p) -> abs (-log p - ent))  
        //printfn "%A" sm
        //what does the above line of code do?
        //Answer: it sorts the probabilities by the absolute value of the difference between the log of the probability and the entropy of the probabilities (i.e. it sorts the probabilities by how different they are from the average probability)
      
        if k = 0 then getTopProbs p sorted 
        else sorted |> Array.takeOrMax k |> getTopProbs p 