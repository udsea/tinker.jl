mutable struct Value
    data::Float32
    grad::Float32
    operation::Symbol
    prev::Vector{Value}
    backward::Function
end