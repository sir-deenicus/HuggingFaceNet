// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open BlingFire
open Microsoft.ML.OnnxRuntime
 
[<EntryPoint>]
let main argv =
    //let robertaQA = new InferenceSession(@"D:\Downloads\NeuralNets\roberta-base-squad2\roberta-base-squad2.onnx", SessionOptions.MakeSessionOptionWithCudaProvider())
    let t5SummarizerEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-base-finetuned-summarize-news\t5-base-summarize-encoder.onnx", SessionOptions.MakeSessionOptionWithCudaProvider())
    
    let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [[100; 19; 3; 9; 794; 1]])
     
    
    let outputs = t5SummarizerEncoder.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]
    
    printfn "%A" ((Seq.head outputs).AsTensor<float32>())
    Console.ReadLine()
    0 // return an integer exit code