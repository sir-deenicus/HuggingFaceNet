open System.Collections.Generic
open System
//Create BPE tokenizer

(*Python version to convert from:

def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word*)

(*def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
    *)

// with open(merges_file, encoding="utf-8") as merges_handle:
//             bpe_merges = merges_handle.read().split("\n")[1:-1]
//         bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
//         self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

//In python, givel l = [1,2,3], what will [1:-1] return?
//Answer: [2]
//What does it mean?
//Answer: It means that it will return all elements of the list except the first and last elements
//How do I write this in F#?
//Answer: l.[1..^1]
//How about using skip and take?
//Answer: l |> Seq.skip 1 |> Seq.take (l.Length - 2)

//In Python what does split() do?
//Answer: It splits a string into a list of strings based on the whitespace characters
//Does it take any parameters?
//Answer: Yes, it takes a parameter called sep which is the separator. If sep is not specified or is None, any whitespace string is a separator and empty strings are removed from the result.
//What counts as whitespace?
//Answer: Whitespace characters are space, tab, linefeed, return, formfeed, and vertical tab.
//What's a regex for that?
//Answer: \s
//self.cache = {}

//in pytorch if t is a tensor of type int64, how to tensor cast to an int type?
//Answer: t.int()



//Answer: torch.tensor([1,2,3], dtype=torch.int64)
//Answer: tensor.type(torch.LongTensor)

let selfCache = new Dictionary<string, string>()
let matchpattern = """'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
 
let buildBPE_merges (path:string) =
    let rawBPEmerges = IO.File.ReadAllLines(path)[1..^1]
    let bpeMerges = 
        [|for merge in rawBPEmerges do
            let splitMerge = merge.Split(' ')
            yield (splitMerge.[0], splitMerge.[1])|]

    let bpeRanks = 
        [|for i in 0..bpeMerges.Length-1 do
            yield (bpeMerges.[i], i)|]

    bpeRanks |> Map.ofArray, bpeMerges |> Map.ofArray


 
let getPairs (word: string) =
    let pairs = new HashSet<_>()
    let mutable prevChar = string word.[0]
    for i in 1..word.Length-1 do
        pairs.Add((string prevChar, string word.[i])) |> ignore
        prevChar <- string word.[i]
    pairs

let bpe (token: string) (bpeRanks: Map<string*string, int>) (bpeMerges: Map<string*string, string>) =
    let word = token 
    let mutable pairs = getPairs token
    if pairs.Count = 0 then token
    else
        let mutable newWord = word
        let mutable exitLoop = false
        while (not exitLoop) do
            let bigram = pairs |> Seq.minBy (fun pair -> bpeRanks.[pair])
            if not (bpeRanks.ContainsKey bigram) then exitLoop <- true
            else
                let first, second = bigram
                let mutable i = 0
                let mutable newWordList = new List<char>()
                while i < newWord.Length do
                    let j = newWord.IndexOf(first, i) 
                    if j = -1 then
                        newWordList.AddRange(newWord.[i..^1])
                        exitLoop <- true 
                    else
                        newWordList.AddRange(newWord.[i..j-1])
                        i <- j
                    if string newWord.[i] = first && i < newWord.Length - 1 && string newWord.[i+1] = second then 
                        newWordList.Add(bpeMerges.[bigram].[0])
                        i <- i + 2
                    else
                        newWordList.Add(newWord.[i])
                        i <- i + 1
                newWord <- newWordList |> Seq.map string |> String.concat "" 
                if newWord.Length = 1 then exitLoop <- true
                else pairs <- getPairs newWord

        newWord
        

