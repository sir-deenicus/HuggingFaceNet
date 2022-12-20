module TensorNet.NLP

open Tokenizers

type Classifier(tokenizer : Tokenizers.GeneralTokenizer, encoder : ONNX.NNet<float32>) =
    member __.X = ()

 
type EncoderDecoder(encoder:ONNX.NNet<float32>, decoder : ONNX.NNet<float32>, tokenizer:Tokenizers.GeneralTokenizer) =
    let decodingStartToken = defaultArg tokenizer.DecoderStartToken 0
    let stopToken = tokenizer.StopToken
    let encoderDecoderModel = new ONNX.EncoderDecoderSampler(encoder, decoder, decodingStartToken, stopToken, ?sentenceEndTokens = tokenizer.EndOfSentenceTokens)
     
    let processContext (definition:string) (context:string) (contextname:string option) (question:string option) =
            let contextwindow = tokenizer.MaxContextWindowLen
            let contextname = defaultArg contextname "Input"
            let question = match question with
                           | Some q -> $"Question: {q} | "
                           | None -> ""
            
            let splitSentences = ResizeArray()
            let rec runRec (context:string) =
                let fullstr = $"Definition: {definition} | {contextname}: {context} | {question}Output: " 
                let tokens = tokenizer.RawTokenize(fullstr)
                if tokens.Length > contextwindow then
                    if splitSentences.Count > 0 then
                        splitSentences.RemoveAt(splitSentences.Count - 1)
                    else
                        splitSentences.AddRange(Tokenizers.splitBySentences context)
                        //remove the last sentence
                        splitSentences.RemoveAt(splitSentences.Count - 1)
                    
                    let context = String.concat " " splitSentences
                    runRec context
                else
                    tokens

            runRec context

    //constructor for when file paths are given.
    new(encoderPath:string, decoderPath:string, tokenizer:Tokenizers.GeneralTokenizer) =
        new EncoderDecoder(new ONNX.NNet<float32>(encoderPath), new ONNX.NNet<float32>(decoderPath), tokenizer)

    //constructor for when 1 file path is given. let's also add a isQuantized flag that will append "-quantized" to the file name
    new(modelPath:string, tokenizer:Tokenizers.GeneralTokenizer, ?isQuantized:bool) = 
        let isQuantized = defaultArg isQuantized false
        let quantized = if isQuantized then "-quantized" else ""
        new EncoderDecoder(new ONNX.NNet<float32>(modelPath + "-encoder" + quantized + ".onnx"), new ONNX.NNet<float32>(modelPath + "-decoder" + quantized + ".onnx"), tokenizer)
         
    
    //let's add a method that takes a string and returns a string
    member this.Run(input:string, ?countBySentences:bool, ?minlen:int, ?maxlen:int, ?sampler) =
        let tokens = tokenizer.RawTokenize(input)
        let res = encoderDecoderModel.Run(tokens, ?countBySentences = countBySentences, ?minlen = minlen, ?seqlen = maxlen, ?sampler = sampler)
        tokenizer.Detokenize(res.Tokens)

    //now we want a version of run that can interleave a definition, context, and Output: prompt
    member this.Run(definition:string, context:string, ?contextname, ?question, ?countBySentences:bool, ?minlen:int, ?maxlen:int, ?sampler) =

        let tokens = processContext definition context contextname question
        let res = encoderDecoderModel.Run(tokens, ?countBySentences = countBySentences, ?minlen = minlen, ?seqlen = maxlen, ?sampler = sampler)
        tokenizer.Detokenize(res.Tokens) 
        
    member this.ProcessContext(definition:string, context:string, ?contextname, ?question) =
        processContext definition context contextname question

    //expose the encoder/decoder  
    member this.EncoderDecoderModel = encoderDecoderModel

    member this.RunEncoder(tokens:Tokenized<int>) =
        encoderDecoderModel.RunEncoder(tokens) 

    //version with strings
    member this.RunEncoder(input:string) =
        let tokens = tokenizer.Tokenize(input)
        encoderDecoderModel.RunEncoder(tokens)

    //version that takes a definition, context, and question
    member this.RunEncoder(definition:string, context:string, ?contextname, ?question) =
        let tokens = processContext definition context contextname question
        encoderDecoderModel.RunEncoder(Tokenized.ofArray 1 tokens)

    member this.RunDecoderStep(tokens) =
        encoderDecoderModel.RunDecoderStep(tokens)

    //Let's make this disposable
    interface System.IDisposable with
        member this.Dispose() =
            encoderDecoderModel.Encoder.Dispose()
            encoderDecoderModel.Decoder.Dispose()