#I @"C:\Users\cybernetic\.nuget\packages\"
#I @"C:\Users\cybernetic\source\repos\" 
#r @"BlingFire\bin\Release\net5.0\BlingFire.dll"
#r @"Prelude\Prelude\bin\Release\net5\Prelude.dll"
#r @"..\bin\x64\Debug\net5.0\TensorNet.dll"
#r @"mathnet.numerics\5.0.0\lib\net5.0\MathNet.Numerics.dll"
#r @"mathnet.numerics.fsharp\5.0.0\lib\net5.0\MathNet.Numerics.FSharp.dll"
#r @"D:\Downloads\NeuralNets\onnx1.11\Microsoft.ML.OnnxRuntime.dll"  
#r @"newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"
#r @"C:\Users\cybernetic\source\repos\SentencePieceWrapper\SentencePieceDotNET\bin\x64\Release\net5.0\SentencePieceDotNET.dll" 

#time "on"

open TensorNet 
open TensorExtensions
open System
open Prelude.Common
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.MatrixExtensions
open Newtonsoft.Json 

#r @"C:\Users\cybernetic\source\repos\Project.Int.Aug\Lucid.TextAnalysis\bin\x64\Debug\net50\Lucid.TextAnalysis.dll"

Tokenizers.initBlingFire @"D:\Downloads\NeuralNets\blingfire\blingfiretokdll.dll"

let t5tokenizer = Tokenizers.SentencePieceTokenizer.NewT5Tokenizer(@"D:\Downloads\NeuralNets\unifiedqa-v2-t5-small\spiece.model")

t5tokenizer.Tokenize "help"

let bertTokenizer = Tokenizers.BlingFireTokenizer.NewBertTokenizer(@"D:\Downloads\NeuralNets\blingfire\")

let robertaTokenizer = Tokenizers.BlingFireTokenizer.NewRobertaTokenizer(@"D:\Downloads\NeuralNets\blingfire\")
 


let str2 = IO.File.ReadAllText (@"D:\Downloads\wikis\corpus.txt")
 
BlingFire.BlingFireUtils.GetWords(str2) |> Seq.toArray |> Array.length

Lucid.TextAnalysis.Tokenizer.splitWordsManualKeepContractions str2 |> Array.length

robertaTokenizer.Tokenizer str2 |> Array.length

String.splitToWordsRegEx str2 |> Array.length
String.splitSentenceManual ("We went to the store. We saw the place.")
[
6625905. / 7471204.
6432559. / 7471204.
5554304. / 7471204.
] |> List.average

[|101; 2057; 2253; 2188; 102|] = Tensor.toArray (bertTokenizer.Tokenize("we went home")) 


let debertatokenizer = Tokenizers.SentencePieceTokenizer.NewDebertaTokenizer(@"D:\Downloads\NeuralNets\nli-deberta-v3-xsmall\spm.model")

let gpt2tokenizer = Tokenizers.BlingFireTokenizer.NewGPT2Tokenizer(@"D:\Downloads\NeuralNets\blingfire\")


robertaTokenizer.Tokenize("we go home") |> Seq.toArray

gpt2tokenizer.Tokenize("we go home") |> Seq.toArray


let _, ts = robertaTokenizer.BatchTokenize(prepend = "Here we are", s = Tokenizers.splitForBatch "There we are")

ts.TokenIds |> Seq.toArray


let model = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\fairseq-dense-355M\fairseq-dense-355M.onnx")

let _, ts = robertaTokenizer.BatchTokenize(prepends = [|"A man is eating"; "We went home"|], s = Tokenizers.splitForBatch "A man eats something. We are partying.")

ts2.AttentionMasks |> Tensor.toArray2D

let ts = gpt2tokenizer.BatchTokenize([|"Hello he said."; "There"|])
ts.TokenIds |> Tensor.toArray2D
let rr = model.RunT(ts.TokenIds, ts.AttentionMasks)
//label_mapping = ['contradiction', 'entailment', 'neutral']
rr[0] |> Tensor.toMatrix










let t5smalldec = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\unifiedqa-v2-t5-small\unifiedqa-v2-t5-small-decoder-quantized.onnx")

t5smalldec.InputKeys