// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open BlingFire

//System.Runtime.InteropServices.NativeLibrary.Load(@"C:\Users\cybernetic\source\repos\BlingFire\bin\Release\net5.0\Files\blingfiretokdll.dll")

let robertaBin = BlingFire.BlingFireUtils.LoadModel  @"roberta.bin"

// Define a function to construct a message to print
let from whom =
    sprintf "from %s" whom

[<EntryPoint>]
let main argv =
    let message = from "F#" // Call the function
    printfn "Hello world %s" message
    0 // return an integer exit code