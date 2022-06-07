///The WSC command does not seem to work. Tested in F# ONNX, Pyhton ONNX and Huggingface pytorch original.
///Relevant ^ https://towardsdatascience.com/t5-a-model-that-explores-the-limits-of-transfer-learning-fb29844890b7
///As with BART summarization has a chance to hallucinate facts or somehow confuse the contexts. 
///Question and Answer however, seems a lot more reliable. And interestingly you can ask: "what is the main idea" and
///the answers for each token section can make for a decent summary. 
///Although the Q/A system sometimes makes things up rather than say nothing (Left brain interpreter...), a fair amount of the time these are promising leads for the actual answer.


#load "tokenizers.fsx"
#load "sampler.fsx"
#load "TestDocSnippets.fsx"
#time "on"
#r @"C:\Users\cybernetic\source\repos\SentencePieceWrapper\SentencePieceDotNET\bin\x64\Release\net5.0\SentencePieceDotNET.dll" 

open Microsoft.ML.OnnxRuntime
open System
open Prelude.Common 
open Prelude.Math 
open ``Tensor-extensions``  
open Tokenizers 
open Sampler

let spiece = new SentencePieceDotNET.SentencePieceDotNET()

spiece.Load(@"D:\Downloads\NeuralNets\xglm-564M\sentencepiece.bpe.model")  
spiece.Encode("私たちは、多くの訪")
let t5TokenizerInfo = {StartToken = None; StopToken = 1}
 
let t5Tokenizer (s: string) = spiece.Encode(s)

let t5detokenizer ids = 
    let ids' = Array.filter ((>) 32_000) ids //remove special tokens which crash our implementation
    if ids'.Length > 0 then spiece.Decode ids'
    else ""

let t5TokenizeSplit maxlen (sents : _ []) =   
    tokenizeSplit (tokenize t5TokenizerInfo t5Tokenizer) maxlen sents

/////////////////////////////////
initBlingFire()

let str = IO.File.ReadAllText (pathCombine DocumentsFolder "doc2.txt")

let sents = BlingFireUtils.splitSentence str 

let t5tokens = t5TokenizeSplit 512 sents

#r "nuget: FsPickler, 5.3.2"
open MBrace.FsPickler


let pickler = FsPickler.CreateBinarySerializer()

let text = IO.File.ReadAllLines (@"C:\Users\cybernetic\Documents\cliptext-text.txt")
let str2 = IO.File.ReadAllText (@"D:\Downloads\wikis\corpus.txt")

let str = text |> String.concat "\n"

(t5Tokenizer str |> Array.length) * sizeof<int>

str.Length * sizeof<string>

let b = pickler.Pickle(t5Tokenizer str)
//write to documents
IO.File.WriteAllText(@"C:\Users\cybernetic\Documents\str1ds.txt", str)
//serialize with fspickler


228_017. / 255_340.

List.average [219705.;228_017.] / 290845.


6_625_905. / 7_471_204.

//let sess =new SessionOptions() 
//.AppendExecutionProvider_DML(0) //0 = inbuilt GPU; 1 = nvidia GPU

///////////////////////////

let ss = SessionOptions.MakeSessionOptionWithCudaProvider()

let t5smallModelEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-small\t5-small-encoder-quantized.onnx")
let t5smallModelDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-small\t5-small-decoder-quantized.onnx")

//Has a tendency to get stuck in a loop
let t5SummarizerEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-base-finetuned-summarize-news\t5-base-summarize-encoder.onnx")
let t5SummarizerDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-base-finetuned-summarize-news\t5-base-summarize-decoder.onnx")

let t5largeEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-large\t5-large-encoder.onnx")
let t5largeDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-large\t5-large-decoder.onnx")

let t5baseEncoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-base\t5-base-encoder-quantized.onnx")
let t5baseDecoder = new InferenceSession(@"D:\Downloads\NeuralNets\t5-base\t5-base-decoder-quantized.onnx")

////////////////////////////////////

let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|t5tokens.[0]|])
let t = Tensors.ArrayTensorExtensions.ToTensor(array2D (t5TokenizeSplit 512 [|"This is a test"|]))

Tensor.toArray2D t

t5detokenizer t5tokens.[1]

t5largeEncoder.Dispose()

let outputs = t5SummarizerEncoder.Run [|NamedOnnxValue.CreateFromTensor("input_ids", t)|]

let encoderStates = Seq.head outputs

encoderStates.Name <- "encoder_hidden_states"
   
outputs.Dispose()

encoderStates.Dispose()  
 
encoderStates.AsTensor<float32>() |> Tensor.toMatrix 

//////////////////////////////

let tokenizeforT5Command command (context : string) =
    let contextCommand = t5Tokenizer (command+":")
     
    context
    |> BlingFireUtils.splitSentence
    |> t5TokenizeSplit (512 - contextCommand.Length)
    |> Array.map (fun c -> Array.concat [contextCommand; c])
 
let tokenizeforT5Summary (context : string) =
    
    let contextCommand = t5Tokenizer "summarize:"
    
    context
    |> BlingFireUtils.splitSentence
    |> t5TokenizeSplit (512 - contextCommand.Length)
    |> Array.map (fun c -> Array.concat [contextCommand; c])

let tokenizeforT5QA (question : string) (context : string) =
    let q =
        t5Tokenizer ("question: " + question)
        |> Array.takeOrMax 511
    
    let contextCommand = t5Tokenizer "context:"
    
    context
    |> BlingFireUtils.splitSentence
    |> t5TokenizeSplit (512 - q.Length - contextCommand.Length)
    |> Array.map (fun c -> Array.concat [q; contextCommand; c])

/////////////////////////////

let testWordsSplitT5 (s:string) =
    {|Token = t5Tokenizer s |> Array.length; 
      Words = Strings.splitToWordsRegEx s |> Array.length |}

let c = t5Tokenizer "This is a test"
let q = t5Tokenizer "summarize: "

t5Tokenizer str
t5Tokenizer "question"
t5Tokenizer "apple"

t5Tokenizer "question apple" |> t5detokenizer
 
Array.append q c = t5Tokenizer "summarize: This is a test"

t5Tokenizer "summarize: This is a test" = (Array.head (tokenizeforT5Summary "This is a test")).[..^1]
 
tokenizeforT5QA "This is a house" "Where are we ?"

////////////////////

let summ = nnSampler t5SummarizerEncoder t5SummarizerDecoder t5detokenizer t5TokenizerInfo.StopToken (Array.argmaxf32) 150 t5tokens

let summ = nnSampler t5smallModelEncoder t5smallModelDecoder t5detokenizer t5TokenizerInfo.StopToken (basicSampler 0 0.9) 200 t5tokens

let summ = nnSampler t5largeEncoder t5largeDecoder t5detokenizer t5TokenizerInfo.StopToken (Array.argmaxf32 ) 200 t5tokens

let summ = nnSampler t5baseEncoder t5baseDecoder t5detokenizer t5TokenizerInfo.StopToken (basicSampler 0 0.9) 200 t5tokens


t5SummarizerDecoder.Dispose()

let str2 = 
    summ |> String.concat "\n"


let t5tokens = t5TokenizeSplit 512 [|"sst2 sentence: Everything was going great until I took an arrow to the knee."|]

let t5tokens = tokenizeforT5QA str "What is the main idea?"

t5tokens.Length

let str = "I tried to teach a class of undergrads about max. likelihood and max. a-posteriori estimation with an example of finding my favorite pizza in New York! Say my best friends lined up a tasting session for my birthday..."

let t5tokens = tokenizeforT5Summary str

testWordsSplitT5 str

[664./895.; 7345./10987.]

///RTE, MNLI, QNLI, SST2, CB and of course question answering, are the most useful tasks. Sentiment might do in a pinch. Summary is middling.
///WSC does not work. Translation is too limited and I'm not sure I see the point of COPA.
let rec recurseOnContext encoder decoder sampler maxlen tokenizer tokens =
    seq {
        let summ =
            nnSampler t5largeEncoder t5largeDecoder t5detokenizer t5TokenizerInfo.StopToken sampler maxlen tokens

        let str2 = summ |> String.concat "\n"

        yield summ

        if summ.Length > 1 then
            yield! recurseOnContext encoder decoder sampler maxlen tokenizer (tokenizer str2)
    }

recurseOnContext t5smallModelEncoder t5smallModelDecoder Array.argmaxf32 200 (tokenizeforT5QA "what is the main issue?") (tokenizeforT5QA "what is the main issue?" str)