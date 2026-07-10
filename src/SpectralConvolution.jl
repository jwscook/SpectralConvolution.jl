module SpectralConvolution

using LinearAlgebra, FFTW
export FourierField, hatvalue, reconstruct, convolve, isaxiscontinuous, detectcontinuity

fftmodetoindex(m::Integer, N::Integer) = m >= 0 ? m + 1 : N + m + 1

struct FourierField{D,T,U,A<:AbstractArray{T,D},B<:AbstractArray{<:Complex{T},D}}
  lims::NTuple{D,NTuple{2,T}}
  NGs::NTuple{D,Int}
  k0s::NTuple{D,U}
  origin::NTuple{D,T}
  values::A
  hatvalues::B
end

"""
    isaxiscontinuous(f, lims, axis, NGtest=8; atol=0.0, rtol=1e-6)

Test whether `f` is CONTINUOUS across the wrap-around boundary along
`axis` — i.e. whether its periodic extension has no jump — by comparing
samples at the two opposite faces `x[axis]=lims[axis][1]` and
`x[axis]=lims[axis][2]`, over a grid of `NGtest` cell-centered points
along every OTHER axis.

This single generic implementation (parameterized by `D` via `lims`'s
type) naturally reduces to:
  - 1D: a single comparison of the two endpoint values.
  - 2D: comparison along a line of points on the opposite EDGES.
  - 3D: comparison over a grid of points on opposing FACES.

This is a NECESSARY, not sufficient, test — it only checks value-matching
at a finite sample of points, so it can miss non-continuity that happens
to coincide at those specific samples.
"""
function isaxiscontinuous(f::F, lims::NTuple{D,NTuple{2,Real}}, axis::Int,
    NGtest=ntuple(_ -> 8, D); atol=0.0, rtol=sqrt(eps())) where {D,F}
  @assert 1 <= axis <= D

  otherpositions = ntuple(D) do i
    i == axis && return Float64[]
    lo, hi = Float64.(lims[i])
    collect(lo .+ ((0:NGtest[i]-1) .+ 0.5) ./ NGtest[i] .* (hi - lo))
  end

  loval, hival = lims[axis]
  scandims = ntuple(i -> i == axis ? 1 : NGtest[i], D)

  for ci in CartesianIndices(scandims)
    xlo = ntuple(i -> i == axis ? loval : otherpositions[i][ci[i]], D)
    xhi = ntuple(i -> i == axis ? hival : otherpositions[i][ci[i]], D)
    isapprox(f(xlo), f(xhi); atol=atol, rtol=rtol) || return false
  end
  return true
end

"""
    detectcontinuity(f, lims, NGtest=8; atol=0.0, rtol=1e-6)

Returns an `NTuple{D,Bool}` suitable for `FourierField`'s `continuous`
keyword, by calling `isaxiscontinuous` independently on each axis.
"""
function detectcontinuity(f::F, lims::NTuple{D,NTuple{2,Real}},
        NGtest=ntuple(_ -> 8, D); atol=0.0, rtol=sqrt(eps())) where {D,F}
  return ntuple(i -> isaxiscontinuous(f, lims, i, NGtest; atol=atol, rtol=rtol), D)
end

"""
    FourierField(f, lims, NG; continuous=ntuple(_->false, length(lims)))

Samples `f` on `NG` cell-centered points per axis over the TRUE domain
`lims`.

For axes where `continuous[i]==false` (the default), the axis is mirrored
at its edges (doubling that axis's length) to remove the boundary value
jump. This does not remove a derivative jump — convergence improves from
O(1/q) to O(1/q^2), not to exponential. That axis's fundamental is then
`k0 = π/(hi-lo)`, the fundamental of the REFLECTED (doubled) domain — FFT
bin `n` corresponds directly to physical mode `n`, no extra factor of 2.
(Restricting to even bins only would keep just the harmonics that are
periodic in the true domain, silently discarding the odd modes that carry
the boundary-reflection correction — exactly what you need to represent a
non-continuous function.)

For axes where `continuous[i]==true`, `f` is assumed genuinely continuous
across the wrap-around boundary along that axis: NO reflection is applied
(that axis's length is NOT doubled), and its fundamental is the TRUE
`k0 = 2π/(hi-lo)`. Use `detectcontinuity` to estimate this automatically.

`hatvalue` needs no per-axis special-casing either way — `Ls[i]` (the
actual stored array length along axis `i`) is already doubled or not as
appropriate, and bin `n` always corresponds directly to mode `n`.

The reflected array is FFT'd once here (`hatvalues`), so `hatvalue` becomes
a plain array lookup rather than a summation.
"""
FourierField(f::F, lims, NG::Integer; kwargs...) where F =
  FourierField(f, lims, ntuple(_ -> NG, length(lims)); kwargs...)
function FourierField(f::F, lims, NGs; continuous=detectcontinuity(f, lims, NGs)) where F
  D = length(lims)
  lims = ntuple(i -> (Float64(lims[i][1]), Float64(lims[i][2])), D)
  NGs = Tuple(NGs)
  continuous = ntuple(i -> isone(NGs[i]) ? true : continuous[i], D)  # never reflect/double a degenerate axis
  @assert length(NGs) == D
  @assert length(continuous) == D
  k0s = ntuple(i -> (continuous[i] ? 2π : π) / (lims[i][2] - lims[i][1]), D)
  @assert all(isfinite, k0s)

  trueposition = ntuple(D) do i
    lo, hi = lims[i]
    collect(lo .+ ((0:NGs[i]-1) .+ 0.5) ./ NGs[i] .* (hi - lo))
  end

  origin = ntuple(i -> trueposition[i][1], D)
  T = typeof(f(origin))
  truevalues = Array{T}(undef, NGs...)
  for ci in CartesianIndices(NGs)
    x = ntuple(i -> trueposition[i][ci[i]], D)
    truevalues[ci] = f(x)
  end

  values = truevalues
  for i in 1:D
    continuous[i] && continue
    values = cat(values, reverse(values; dims=i); dims=i)
  end
  hatvalues = fft(values) ./ prod(size(values))

  return FourierField(lims, NGs, k0s, origin, values, hatvalues)
end

function hatvalue(ff::FourierField{D}, modes::NTuple{D,<:Integer}) where D
  Ls = size(ff.hatvalues)   # ACTUAL (per-axis reflected or not) array length
  idx = ntuple(D) do i
    bin = modes[i]
    if Ls[i] == 1   # degenerate axis: only the DC mode exists
      @assert bin == 0 "axis $i is degenerate (zero extent, lims[1]==lims[2]): only mode 0 is defined, got $bin"
      return 1
    end
    @assert abs(bin) <= Ls[i] ÷ 2 - 1 "mode $(modes[i]) on axis $i exceeds the representable Nyquist range of $((Ls[i]÷2 - 1))"
    fftmodetoindex(bin, Ls[i])
  end
  return ff.hatvalues[idx...]
end

function reconstruct(ff::FourierField, x, maxshells; atol=0.0, rtol=sqrt(eps()))
  return convolve(ff, z->1, x, 0, maxshells; atol=atol, rtol=rtol)
end

function convolve(ff::FourierField{D}, g::G, x, k, maxshells;
    atol=0.0, rtol=sqrt(eps())) where {D,G}
  length(x) == D || error("x must have length $D")
  length(maxshells) == D || error("maxshells must have length $D")
  maxshells = ntuple(i -> isone(ff.NGs[i]) ? 0 : maxshells[i], D)  # a degenerate axis only ever has mode 0
  output = hatvalue(ff, ntuple(_ -> 0, D)) * g(k)
  oldnorm = norm(output)
  for shell in 1:maximum(maxshells)
    smax = ntuple(i -> min(shell, maxshells[i]), D)
    shellsum = zero(output)
    for ci in CartesianIndices(ntuple(i -> -smax[i]:smax[i], D))
      modes = Tuple(ci)
      maximum(abs.(modes)) == shell || continue
      q = modes .* ff.k0s
      val = hatvalue(ff, modes) * g(k .- q) * cis(sum(q .* (x .- ff.origin)))
      shellsum += val
    end
    output += shellsum
    newnorm = norm(output)
    isapprox(newnorm, oldnorm; atol=atol, rtol=rtol) && break
    oldnorm = newnorm
  end
  return output
end

end # module SpectralConvolution
