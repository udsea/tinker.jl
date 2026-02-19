using SpecialFunctions # For erf()

# --- Core Engine ---
mutable struct Value
    v::Float32           # Scalar Value
    Δ::Float32           # Gradient (Delta)
    op::Symbol           # Operation tracker
    prev::Vector{Value}  # Ancestors in the graph
    backward!::Function  # Closure to compute local gradient

    function Value(v; op=:leaf, prev=Value[])
        new(Float32(v), 0.0f0, op, prev, () -> nothing)
    end
end

function Base.show(io::IO, x::Value)
    print(io, "Value(v=$(round(x.v, digits=4)), Δ=$(round(x.Δ, digits=4)), op=$(x.op))")
end

# --- Basic Arithmetic & Overloads ---
import Base: +, *, ^, -, /, tanh

function +(a::Value, b::Value)
    out = Value(a.v + b.v, op=:+, prev=[a, b])
    out.backward! = () -> (a.Δ += out.Δ; b.Δ += out.Δ)
    return out
end

function *(a::Value, b::Value)
    out = Value(a.v * b.v, op=:*, prev=[a, b])
    out.backward! = () -> (a.Δ += b.v * out.Δ; b.Δ += a.v * out.Δ)
    return out
end

function ^(a::Value, n::Real)
    out = Value(a.v^n, op=:^, prev=[a])
    out.backward! = () -> (a.Δ += (n * a.v^(n-1)) * out.Δ)
    return out
end

# Handle Scalars (Promotion)
Base.:+(a::Value, b::Real) = a + Value(b)
Base.:+(a::Real, b::Value) = Value(a) + b
Base.:*(a::Value, b::Real) = a * Value(b)
Base.:*(a::Real, b::Value) = Value(a) * b
Base.:-(a::Value) = a * -1.0f0
Base.:-(a::Value, b::Value) = a + (-b)
Base.:-(a::Value, b::Real) = a + (-b)
Base.:/(a::Value, b::Value) = a * (b^-1.0f0)
Base.:/(a::Value, b::Real) = a * (Value(b)^-1.0f0)

# --- Activations ---

# Sigmoid: s(x) = 1 / (1 + exp(-x))
function σ(x::Value)
    s = 1.0f0 / (1.0f0 + exp(-x.v))
    out = Value(s, op=:sigmoid, prev=[x])
    out.backward! = () -> (x.Δ += (s * (1.0f0 - s)) * out.Δ)
    return out
end

# ReLU
function Relu(x::Value)
    out = Value(max(0.0f0, x.v), op=:ReLU, prev=[x])
    out.backward! = () -> (x.Δ += (x.v > 0 ? 1.0f0 : 0.0f0) * out.Δ)
    return out
end

# Tanh
function tanh(x::Value)
    t = Base.tanh(x.v)
    out = Value(t, op=:tanh, prev=[x])
    out.backward! = () -> (x.Δ += (1.0f0 - t^2) * out.Δ)
    return out
end

# Leaky ReLU
function leaky_relu(x::Value, α=0.01f0)
    out = Value(x.v > 0 ? x.v : α * x.v, op=:lrelu, prev=[x])
    out.backward! = () -> (x.Δ += (x.v > 0 ? 1.0f0 : α) * out.Δ)
    return out
end

# GELU
function gelu(x::Value)
    cdf = 0.5f0 * (1.0f0 + SpecialFunctions.erf(x.v / sqrt(2.0f0)))
    pdf = (1.0f0 / sqrt(2.0f0 * π)) * exp(-0.5f0 * x.v^2)
    out = Value(x.v * cdf, op=:gelu, prev=[x])
    out.backward! = () -> (x.Δ += (cdf + x.v * pdf) * out.Δ)
    return out
end

# --- The Backprop Trigger ---
function ∇(root::Value)
    topo = Value[]
    visited = Set{Value}()
    function build_topo(v)
        if v ∉ visited
            push!(visited, v)
            for child in v.prev
                build_topo(child)
            end
            push!(topo, v)
        end
    end
    build_topo(root)
    
    # Zero out grads before backprop
    for node in topo; node.Δ = 0.0f0; end
    root.Δ = 1.0f0

    for node in reverse(topo)
        node.backward!()
    end
end

# --- Example ---
begin
    x = Value(0.0f0)
    y = sigmoid(x) # Should be 0.5
    ∇(y)
    println("Sigmoid(0): ", y.v)
    println("Gradient at 0: ", x.Δ) # Should be 0.25 (0.5 * (1-0.5))
end