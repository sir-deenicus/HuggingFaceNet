module TensorNet.TensorExtensions

open System.Collections.Generic
open Microsoft.ML.OnnxRuntime
open System
open MathNet.Numerics.LinearAlgebra

module Array =
    let shape (a: 'a[,]) = [|Array2D.length1 a; Array2D.length2 a|]

    let toTensor (a : 'a []) = Tensors.ArrayTensorExtensions.ToTensor(a)
 
    let argmax minvalue (d:_[]) =
        let mutable topIndex = 0
        let mutable topScore = minvalue

        for i in 0..d.Length - 1 do
            if d.[i] > topScore then 
                topScore <- d.[i]
                topIndex <- i    
        topIndex, topScore

    let argmaxf32 d = argmax Single.MinValue d
     
module Array2D =
    let shape (a: 'a[,]) = [|Array2D.length1 a; Array2D.length2 a|]

    let toTensor (a : 'a [,]) = Tensors.ArrayTensorExtensions.ToTensor(a)
    
module Array3D =
    let ofArray (a: _ [] [] []) = 
        let d, w, h = a.Length, a[0].Length, a.[0].[0].Length
        
        Array3D.init d w h (fun i j k -> a[i].[j].[k])

    let shape (a : 'a [,,]) = [|Array3D.length1 a; Array3D.length2 a; Array3D.length3 a|]

    let toTensor (a : 'a [,,]) = Tensors.ArrayTensorExtensions.ToTensor(a)

module Array4D =
    let ofArray (a: _ [] [] [] []) = 
        let b, d, w, h = a.Length, a[0].Length, a.[0].[0].Length, a.[0].[0].[0].Length
        
        Array4D.init b d w h (fun i j k l -> a[i].[j].[k].[l])

    let shape (a : 'a [,,,]) = [|Array4D.length1 a; Array4D.length2 a; Array4D.length3 a; Array4D.length4 a|]

    let toTensor (a : 'a [,,,]) = Tensors.ArrayTensorExtensions.ToTensor(a)
      

type Tensors.Tensor<'a> with  
    member t.GetSlice(r:int,startIdx2:int option, endIdx2 : int option) =
            let dim = t.Dimensions.ToArray()
            if dim.Length <> 2 then failwith "Dimensions must be = 2"

            let sidx2 = defaultArg startIdx2 0
            let endidx2 = defaultArg endIdx2 (dim.[1] - 1) 
            [|for c in sidx2..endidx2  -> t.[r, c]|]

    member t.GetSlice(startIdx1:int option, endIdx1 : int option, c:int) =
        let dim = t.Dimensions.ToArray()
        if dim.Length <> 2 then 
            failwith "Dimensions must be = 2"

        let sidx1 = defaultArg startIdx1 0
        let endidx1 = defaultArg endIdx1 (dim.[0] - 1) 
        [|for r in sidx1..endidx1 -> t.[r,c]|]  
    
    member t.GetSlice(startIdx1:int option, endIdx1 : int option,startIdx2:int option, endIdx2 : int option) =
        let dim = t.Dimensions.ToArray()
        if dim.Length <> 2 then failwith "Dimensions must be = 2"

        let sidx1 = defaultArg startIdx1 0
        let sidx2 = defaultArg startIdx2 0
        let endidx1 = defaultArg endIdx1 (dim.[0] - 1)
        let endidx2 = defaultArg endIdx2 (dim.[1] - 1)
        [| for r in sidx1..endidx1 -> [|for c in sidx2..endidx2 -> t.[r,c]|]|]
 
    member t.GetSlice(m:int,r:int, startIdx3:int option, endIdx3 : int option) =
        let dim = t.Dimensions.ToArray()
        if dim.Length <> 3 then failwith "Dimensions must be = 3"

        let sidx3 = defaultArg startIdx3 0
        let endidx3 = defaultArg endIdx3 (dim.[2] - 1) 
        [|for c in sidx3..endidx3 -> t.[m,r, c]|]  
        
    member t.GetReverseIndex(i, offset) = 
        let d = t.Dimensions.ToArray()
        int d.[i] - 1 - offset

    member t.GetSlice(m:int,startIdx2:int option, endIdx2 : int option,startIdx3:int option, endIdx3 : int option) =
        let dim = t.Dimensions.ToArray()
        if dim.Length <> 3 then failwith "Dimensions must be = 3"

        let sidx2 = defaultArg startIdx2 0
        let endidx2 = defaultArg endIdx2 (dim.[1] - 1)
        let sidx3 = defaultArg startIdx3 0
        let endidx3 = defaultArg endIdx3 (dim.[2] - 1)
        [|for r in sidx2..endidx2 -> [|for c in sidx3..endidx3  -> t.[m,r, c]|]|]

    member t.GetSlice(startIdx1:int option, endIdx1:int option, startIdx2:int option, endIdx2 : int option, k:int) =
        let dim = t.Dimensions.ToArray()
        if dim.Length <> 3 then failwith "Effective Dimensions must be = 3"

        let sidx1 = defaultArg startIdx1 0
        let endidx1 = defaultArg endIdx1 (dim.[0] - 1)
        let sidx2 = defaultArg startIdx2 0
        let endidx2 = defaultArg endIdx2 (dim.[1] - 1)
        [|for i in sidx1..endidx1 -> [|for r in sidx2..endidx2 -> t.[i,r, k]|]|]
        

    member t.GetSlice(i:int, startIdx2:int option, endIdx2 : int option, startIdx3:int option, endIdx3 : int option, l : int) =
        let dim = t.Dimensions.ToArray()
        if dim.Length <> 4 then failwith "Dimensions must be = 4"

        let sidx2 = defaultArg startIdx2 0
        let endidx2 = defaultArg endIdx2 (dim.[1] - 1)
        let sidx3 = defaultArg startIdx3 0
        let endidx3 = defaultArg endIdx3 (dim.[2] - 1)
        [|for r in sidx2..endidx2 -> [|for c in sidx3..endidx3 -> t.[i,r, c, l]|]|]  

    member t.Shape = t.Dimensions.ToArray()

    member t.ToOnnxValue(name:string) =
        NamedOnnxValue.CreateFromTensor(name, t)


module Tensor =  
    let shape (t : Tensors.Tensor<_>) = t.Shape
    
    let toOnnxValue tensorname (t: Tensors.Tensor<_>) = NamedOnnxValue.CreateFromTensor(tensorname, t)

    let ofNestedSeq2D s = s |> array2D |> Array2D.toTensor

    let toArray (t: Tensors.Tensor<_>) =
        let dims = t.Dimensions.ToArray()

        match dims with
        | [| d |] -> [| for i in 0..d-1 -> t[i]|]
        | [|1; d|] -> [| for i in 0..d-1 -> t[0, i]|]

        | _ -> failwith "Incompatible dimensions"

    
    let argmax minvalue (t: Tensors.Tensor<_>) =
        let dims = t.Dimensions.ToArray()
        if dims.Length > 1 then failwith "Tensor must be 1D"
        let d = dims[0]
        let mutable topIndex = 0
        let mutable topScore = minvalue

        for i in 0..d - 1 do
            if t[i] > topScore then 
                topScore <- t[i]
                topIndex <- i    
        topIndex, topScore  
    
    let argmaxf32 d = argmax Single.MinValue d 
 
    
    let toJaggedArray2D (t : Tensors.Tensor<_>) =
        let dims = t.Dimensions.ToArray()
        match dims with
        | [| w; h|] ->
            [| for i in 0..w - 1 ->
                [| for j in 0..h - 1 -> t.[i, j] |] |]
        
        | [| 1; w; h |] ->
            [| for i in 0..w - 1 ->
                [| for j in 0..h - 1 -> t.[0, i, j] |] |]
        
        | [| 1; 1; w; h |] ->
            [| for i in 0..w - 1 ->
                [| for j in 0..h - 1 -> t.[0, 0, i, j] |] |]
        | _ -> failwith "Incompatible dimensions"
        
        
    let toJaggedArray3D (t : Tensors.Tensor<_>) =
        let dims = t.Dimensions.ToArray()
        match dims with
        | [| d; w; h |] ->
            [| for i in 0..d - 1 ->
                [| for j in 0..w - 1 ->
                    [| for k in 0..h - 1 -> t.[i, j, k] |] |] |]
                    
        | [| 1; d; w; h |] ->
            [| for i in 0..d - 1 ->
                [| for j in 0..w - 1 ->
                    [| for k in 0..h - 1 -> t.[0, i, j, k] |] |] |] 
                    
        | _ -> failwith "Incompatible dimensions" 


    let toArray2D t = array2D (toJaggedArray2D t) 
    
    let toArray3D (t : Tensors.Tensor<_>) =
        let dims = t.Dimensions.ToArray()
        match dims with
        | [| d; w; h |] ->
            Array3D.init d w h (fun i j k -> t[i,j,k])
        | [|1 ; d; w; h|] -> Array3D.init d w h (fun i j k -> t[0, i,j,k])
            
        | _ -> failwith "Incompatible dimensions"

    let toVector t = vector (toArray t)
    
    let toMatrix t = matrix (toJaggedArray2D t)

    let toDenseVector t = DenseVector.ofArray (toArray t)

    let toDenseMatrix t = DenseMatrix.ofRowArrays (toJaggedArray2D t)
