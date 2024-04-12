using JSON
using Serialization, ArgParse
using CSV, DataFrames, HDF5
include("SpatialRegionTools.jl")


args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--datapath"
            arg_type=String
            default="/data/schestakov/re-identification/data"
        "--filename"
            arg_type=String
            default="own"
        "--parampath"
            arg_type=String
            default="./hyper-parameters_sf.json"
    end
    parse_args(s; as_symbols=true)
end

filename = args[:filename]
datapath = args[:datapath]
parampath = args[:parampath]
param = JSON.parsefile(parampath)
regionps = param["region"]
cityname = regionps["cityname"]
cellsize = regionps["cellsize"]

region = SpatialRegion(cityname,
                       regionps["minlon"], regionps["minlat"],
                       regionps["maxlon"], regionps["maxlat"],
                       cellsize, cellsize,
                       regionps["minfreq"], # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4)

println("Building spatial region with:
        cityname=$(region.name),
        minlon=$(region.minlon),
        minlat=$(region.minlat),
        maxlon=$(region.maxlon),
        maxlat=$(region.maxlat),
        xstep=$(region.xstep),
        ystep=$(region.ystep),
        minfreq=$(region.minfreq)")

paramfile = "$datapath/$(region.name)-param-cell$(Int(cellsize))"
if isfile(paramfile)
    println("Reading parameter file from $paramfile")
    region = deserialize(paramfile)
    println("Loaded $paramfile into region")
else
    println("Cannot find $paramfile")
end

trjfile = "$datapath/$filename.h5"
outfile = "$datapath/$filename.t"
#We readin .h5 file
h5open(trjfile, "r") do f
    # We define this function for later use
    seq2str(seq) = join(map(string, seq), " ") * "\n"
    # We get the amount of trajectories and iterate over them
    num = read(attributes(f)["num"])

    # Open File for writing
    open(outfile, "w") do s
        for i = 1:num
            # We read the trip
            trip = read(f["/trips/$i"])     
            #println(trip)           
            # We need to transfer it to a sequence of cells using our region:
            seq = trip2seq(region, trip)  
            #println(seq2str(seq))      
            # Then tro write we need to put it into seq2string function
            write(s, seq2str(seq))
        end 
    end 
end

