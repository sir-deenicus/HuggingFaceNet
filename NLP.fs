module TensorNet.NLP

type Classifier(tokenizer : Tokenizers.GeneralTokenizer, encoder : ONNX.NNet<float32>) =
    member __.X = ()

type EncoderDecoder(tokenizer : Tokenizers.GeneralTokenizer, encoderdecodernet: ONNX.EncoderDecoderSampler) =
    member __.Query(query:string, text : string[]) =
        
        let tokens = tokenizer.BatchTokenize(query, text)
        tokens.TokensAndMasks
        
