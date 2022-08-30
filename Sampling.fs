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
    StartToken : int 
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

    let getTotProb maxp (ps: _ []) = 
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
        getTotProb maxp ps'

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
        if samplerState.SampleLen >= samplingInfo.BlockStartTokenIndex then
            logits.[samplingInfo.StartToken] <- -infinityf

        //suppress end of sentence tokens while < minsentlen
        if samplerState.SampleLen < samplingInfo.MinSentLen then
            logits.[samplingInfo.StopToken] <- -infinityf

        Array.argmaxf32 logits


    let sample k p samplingInfo samplerState (logits: float32 []) = 
        if samplerState.SampleLen >= samplingInfo.BlockStartTokenIndex then
            logits.[samplingInfo.StartToken] <- -infinityf

        //suppress end of sentence tokens while < minsentlen
        if samplerState.SampleLen < samplingInfo.MinSentLen then
            logits.[samplingInfo.StopToken] <- -infinityf

        let probs =
            [| for i in 0 .. logits.Length - 1 -> i, exp (float32 logits.[i]) |]
            |> Array.normalize

        let choices = getLargeProbItems p probs

        let i =
            if k > 0 then
                choices |> Array.take k |> discreteSampleIndex
            else
                discreteSampleIndex choices

        i, snd probs.[i] 


    let sampleTypical k p samplingInfo samplerState (logits: float32 []) = 
        if samplerState.SampleLen >= samplingInfo.BlockStartTokenIndex then
            logits.[samplingInfo.StartToken] <- -infinityf

        //suppress end of sentence tokens while < minsentlen
        if samplerState.SampleLen < samplingInfo.MinSentLen then
            logits.[samplingInfo.StopToken] <- -infinityf

        let probs =
            [| for i in 0 .. logits.Length - 1 -> i, exp (float32 logits.[i]) |]
            |> Array.normalize

        let ent = -1f * (Array.sumBy (fun (_, p) -> p * log p) probs)
        
        let sorted = probs |> Array.sortBy (fun (_, p) -> abs (-log p - ent))  
        
        let choices = 
            if k = 0 then getTotProb p sorted 
            else sorted |> Array.takeOrMax k |> getTotProb p

        let i = discreteSampleIndex choices

        i, snd probs.[i] 