#I @"C:\Users\cybernetic\.nuget\packages\"
#I @"C:\Users\cybernetic\source\repos\" 
#r @"BlingFire\bin\Release\net5.0\BlingFire.dll"
#r @"Prelude\Prelude\bin\Release\net5\Prelude.dll"
#r @"..\bin\x64\Debug\net5.0\TensorNet.dll"
#r @"mathnet.numerics\5.0.0\lib\net5.0\MathNet.Numerics.dll"
#r @"mathnet.numerics.fsharp\5.0.0\lib\net5.0\MathNet.Numerics.FSharp.dll"
#r @"D:\Downloads\NeuralNets\onnx1.11\Microsoft.ML.OnnxRuntime.dll"  
#r @"newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"

#time "on"

open TensorNet 
open TensorExtensions

open Prelude.Common
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.MatrixExtensions
open Newtonsoft.Json 

let model = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2.onnx")
let model2 = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2-opt.onnx")
let model3 = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2-quantized.onnx")

Tokenizers.initBlingFire @"D:\Downloads\NeuralNets\blingfire\blingfiretokdll.dll"


let bertTokenizer = Tokenizers.BlingFireTokenizer.NewBertTokenizer(@"D:\Downloads\NeuralNets\blingfire\")
 

let corpus = [|
    "A man is eating food.";
    "A man is eating a piece of bread.";
    "The girl is carrying a baby.";
    "A man is riding a horse.";
    "A woman is playing violin.";
    "Two men pushed carts through the woods.";
    "A man is riding a white horse on an enclosed ground.";
    "A monkey is playing drums.";
    "the cat went out and about";
    "A cheetah is running behind its prey.";
    "Crystallographers will be presenting their novel research";
    "The felines are playing";
    "We made a scientific discovery on the nature of the minerals";
    "We found some interesting new facts about these rocks"
|]
        
let l, r = bertTokenizer.BatchTokenize(corpus |> Array.map Array.lift)

l
r.AttentionMasks |> Tensor.toArray2D
r.TokenIds |> Tensor.toArray2D

let res = model.RunT(r.TokenIds, r.AttentionMasks)
let res2 = model2.RunT(r.TokenIds, r.AttentionMasks)
let res3 = model3.RunT(r.TokenIds, r.AttentionMasks)

res[0].Shape
   
let m = res[0] |> Tensor.toMatrix
let m2 = res2[0] |> Tensor.toMatrix
let m3 = res3[0] |> Tensor.toMatrix


m[0, *] *  m[1, *]

m[0, *].Norm(2.)

let sim = m * m.Transpose()
m2 * m2.Transpose()
m3 * m3.Transpose()

(m - m3) |> Matrix.Abs |> Matrix.sum
 
(m - m3) |> Matrix.Abs |> Matrix.toRowArrays |> Array.map Array.average
(m - m3) |> Matrix.Abs |> Matrix.toRowArrays |> Array.concat |> Array.average


Array.zip corpus (sim[12, *].ToArray()) |> Array.sortByDescending snd 