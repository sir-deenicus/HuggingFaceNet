#r "nuget: SharpHash, 1.1.0"
#r "nuget: K4os.Hash.xxHash, 1.0.8"
#r "nuget: System.Data.HashFunction.FNV, 2.0.0"
#r "nuget: System.Data.HashFunction.CityHash, 2.0.0"
#r "nuget: System.Data.HashFunction.MurmurHash, 2.0.0"
//json nuget
#r "nuget: Newtonsoft.Json, 12.0.3"

//prelude
#I @"C:\Users\cybernetic\source\repos\"
#r @"Prelude\Prelude\bin\Release\net5\Prelude.dll"

open System
open Prelude.Common
open System.Data

//to time use timeThis function. Its type is int -> (unit -> 'a) -> 'a

let lookup =
    IO.File.ReadAllText(pathCombine DocumentsFolder "wordnet-text.json")
    |> Newtonsoft.Json.JsonConvert.DeserializeObject<string []>

//Task: You are to search your knowledge of programming to give expert advice on how to improve a program.
//let us improve this code to not be so repetitive. Do you know how?
//Let's think step by step
//What are we repeating alot?
//We are repeating the creation of the hash function, the computation of the hash, and the conversion of the hash to bytes
//We can make a function that takes a hash function and returns a function that takes a string and returns a byte array 
//We can then use this function to map over the array of strings
//code:
let hashToint64 (hasher: SharpHash.Interfaces.IHash) =
    fun s ->
        hasher
            .ComputeString(s, Text.Encoding.UTF8)
            .GetUInt64()    

let timeHash n (hasher: string -> uint64) =
    timeThis n (fun () ->
        lookup
        |> Array.map hasher)

let timeSpanToMs (ts: TimeSpan) = ts.TotalMilliseconds

let getRunStats n hashname hasher =
    let res = timeHash n hasher
    
    {| 
        Hasher = hashname
        Time = List.averageBy timeSpanToMs res.ElapsedTimes
        TotalTime = res.ElapsedTotal
        NoDuplicates = Hashset(res.Result).Count = res.Result.Length
    |}
 
let sharphasherfnv = SharpHash.Base.HashFactory.Hash64.CreateFNV()
let shfnv = getRunStats 200 sharphasherfnv.Name (hashToint64 sharphasherfnv)
shfnv.Time

let sharphasherfnv1a = SharpHash.Base.HashFactory.Hash64.CreateFNV1a()
let shfnv1a = getRunStats 200 sharphasherfnv1a.Name (hashToint64 sharphasherfnv1a)
shfnv1a.Time

let sharphashermurmur2 = SharpHash.Base.HashFactory.Hash64.CreateMurmur2()
let shmurmur2 = getRunStats 200 sharphashermurmur2.Name (hashToint64 sharphashermurmur2)
shmurmur2.Time

let sharphasherxxhash = SharpHash.Base.HashFactory.Hash64.CreateXXHash64()
let shxxhash = getRunStats 200 sharphasherxxhash.Name (hashToint64 sharphasherxxhash)
shxxhash.Time


//k4os is a xxHash
//bytes of lookup[0]
//let bytes = System.Text.Encoding.UTF8.GetBytes(lookup.[0])
//let k4osxxhash = K4os.Hash.xxHash.XXH64.DigestOf(bytes)
//Alright do same for k4os
let k4osxxhasher (s: string) =
    let bytes = System.Text.Encoding.UTF8.GetBytes(s)
    K4os.Hash.xxHash.XXH64.DigestOf(bytes)

let k4osxxhash = getRunStats 200 "k4osxxhash" k4osxxhasher
k4osxxhash.Time


//Now let's do the same for System.Data.HashFunctions
 
//general data.HashFunction hasher has type Data.HashFunction.IHashFunction
let dataHasher (hasher: Data.HashFunction.IHashFunction) (s: string) =
        let bytes = hasher.ComputeHash(System.Text.Encoding.UTF8.GetBytes(s)).Hash
        BitConverter.ToUInt64(bytes, 0)

let fnvhasher = dataHasher (HashFunction.FNV.FNV1Factory.Instance.Create())

let fnvhash = getRunStats 200 "data-fnvhash" fnvhasher
fnvhash.Time

let fnv1ahasher = dataHasher (HashFunction.FNV.FNV1aFactory.Instance.Create())

let fnv1ahash = getRunStats 200 "data-fnv1ahash" fnv1ahasher
fnv1ahash.Time


let murmur2hasher = dataHasher (HashFunction.MurmurHash.MurmurHash2Factory.Instance.Create())

let murmur2hash = getRunStats 200 "data-murmur2hash" murmur2hasher
murmur2hash.Time

//put all the times and names in a list and sort by time
let times = [shfnv; shfnv1a; shmurmur2; shxxhash; k4osxxhash; fnvhash; fnv1ahash; murmur2hash]
let sortedTimes = times |> List.sortBy (fun x -> x.Time)
