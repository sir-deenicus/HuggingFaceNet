// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open BlingFire
open Microsoft.ML.OnnxRuntime
open TensorNet
 
let model = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2.onnx")
let model2 = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2-opt.onnx")
let model3 = new ONNX.NNet<float32>(@"D:\Downloads\NeuralNets\all-MiniLM-L6-v2\all-MiniLM-L6-v2-quantized.onnx")

Tokenizers.initBlingFire @"D:\Downloads\NeuralNets\blingfire\blingfiretokdll.dll" |> ignore

let bertTokenizer = Tokenizers.BlingFireTokenizer.NewBertTokenizer(@"D:\Downloads\NeuralNets\blingfire\")

let docstore = TensorNet.DocumentVectorStore(@"C:\Users\cybernetic\Documents", "wordnetdb")

let sv = TensorNet.SentenceVectors(bertTokenizer, model3, docstore)

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
         



[<EntryPoint>]
let main argv =
    //let robertaQA = new InferenceSession(@"D:\Downloads\NeuralNets\roberta-base-squad2\roberta-base-squad2.onnx", SessionOptions.MakeSessionOptionWithCudaProvider())
    //let t5SummarizerEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-base-finetuned-summarize-news\t5-base-summarize-encoder.onnx", SessionOptions.MakeSessionOptionWithCudaProvider())
    
    //let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [[100; 19; 3; 9; 794; 1]])
     
    
    //let outputs = t5SummarizerEncoder.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]
    
    //printfn "%A" ((Seq.head outputs).AsTensor<float32>())
    sv.Process(corpus |> Array.map (fun s -> {Title = ""; Text = s}))
    sv.GetNeighbors("rocs")
    Console.ReadLine()
    0 // return an integer exit code