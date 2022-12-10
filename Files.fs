module TensorNet.Files

open System.IO

let private fs = MBrace.FsPickler.FsPickler.CreateBinarySerializer()

let load<'a> fname =
    use pstream = File.OpenRead(fname)
    fs.Deserialize<'a>(pstream)

let save fname item =
    use pstream = File.OpenWrite(fname)
    fs.Serialize(pstream, item)
