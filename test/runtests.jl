using ForwardDiff, SpectralConvolution, Test

@testset "SpectralConvolution" begin

@testset "reflection is continuous at junctions" begin
  fU(x) = 0.3 + 0.5*x[1] + 0.2*sin(3*x[1])
  ff = FourierField(fU, ((0.0,1.0),), (16,))
  @test isapprox(ff.values[16], ff.values[17]; atol=1e-12)
  @test isapprox(ff.values[end], ff.values[1]; atol=1e-12)
end

@testset "non-periodic function: value error shrinks with more shells" begin
  fU(x) = 0.3 + 0.5*x[1] + 0.2*sin(3*x[1])
  x0 = (0.6,)
  ff = FourierField(fU, ((0.0, 1.0),), (256,))
  errs = [abs(real(reconstruct(ff, x0, (m,))) - fU(x0)) for m in (4, 8, 16)]
  @test errs[3] < errs[2] < errs[1]
end

@testset "periodic function: reconstruction converges with more shells (not exact at few)" begin
  ϕ = 0.3
  fU(x) = sin(2π*x[1] + ϕ) * sin(2π*x[2]+ϕ)
  ff = FourierField(fU, ((0.0, 1.0), (0.0, 1.0)), (64, 64))
  x = (0.13, 0.71)
  errs = [abs(real(reconstruct(ff, x, (m,m))) - fU(x)) for m in (4, 10, 20)]
  @test errs[3] <= errs[2] <= errs[1]
end

@testset "phase sign sanity" begin
  fU(x) = sin(2π*x[1]) + 0.3*cos(4π*x[1])
  ff = FourierField(fU, ((0.0, 1.0),), (32,))
  for (mode, q) in enumerate((2π, 4π))
    @test isapprox(hatvalue(ff, (mode,)), conj(hatvalue(ff, (-mode,))); atol=1e-10)
  end
end

@testset "convolve, constant U reduces to plain product" begin
  constU(x) = 2.0
  ff = FourierField(constU, ((0.0,1.0),), (16,))
  g(k) = 1/(1+k[1]^2)
  result = convolve(ff, g, (0.3,), (0.7,), (3,))
  @test isapprox(result, 2.0*g((0.7,)); atol=1e-10)
end

@testset "scalar NG dispatch matches tuple NG" begin
  fU(x,y) = sin(2π*x) * cos(2π*y)
  f2(x) = fU(x[1],x[2])
  ff_scalar = FourierField(f2, ((0.0,1.0),(0.0,1.0)), 20)
  ff_tuple  = FourierField(f2, ((0.0,1.0),(0.0,1.0)), (20,20))
  @test ff_scalar.values == ff_tuple.values
  @test ff_scalar.k0s == ff_tuple.k0s
end

@testset "continuous dimension: no reflection, true (unhalved) fundamental" begin
  ϕ = 0.3
  fU(x) = sin(2π*x[1] + ϕ) * (0.3 + 0.5*x[2] + 0.2*sin(3*x[2]))
  lims2 = ((0.0,1.0),(0.0,1.0))
  @test detectcontinuity(fU, lims2) == (true, false)
  ff = FourierField(fU, lims2, (64,64))

  @test size(ff.hatvalues, 1) == 64    # continuous axis: undoubled
  @test size(ff.hatvalues, 2) == 128   # non-continuous axis: doubled
  @test ff.k0s[1] ≈ 2π                 # true fundamental
  @test ff.k0s[2] ≈ π                  # reflected-domain fundamental

  x = (0.4, 0.6)
  errs = [abs(real(reconstruct(ff, x, (m,m))) - fU(x)) for m in (2,8,20)]
  @test errs[3] < errs[2] < errs[1]
end

@testset "degenerate axis: R-phi-Z with no phi variation" begin
  # f only depends on x[1] (R) and x[3] (Z); phi (x[2]) axis is degenerate
  fU(x) = sin(2π*x[1]) * (0.3 + 0.5*x[3] + 0.2*sin(3*x[3]))
  φ0 = 1.234
  lims = ((0.0,1.0), (φ0,φ0), (0.0,1.0))
  ff = FourierField(fU, lims, (32, 5, 32))  # NG for phi axis is ignored/forced to 1

  @test size(ff.hatvalues, 2) == 1
  @test ff.k0s[2] == 0.0
  @test ff.origin[2] == φ0

  x = (0.37, 999.0, 0.62)  # phi entry (999.0) must be irrelevant to the result
  errs = [abs(real(reconstruct(ff, x, (m, 3, m))) - fU((x[1], φ0, x[3]))) for m in (2, 8, 20)]
  @test errs[3] < errs[2] < errs[1]

  # passing nonzero maxshells for the degenerate axis must not error, and
  # must give the same answer as passing 0 explicitly
  r1 = reconstruct(ff, x, (10, 0, 10))
  r2 = reconstruct(ff, x, (10, 7, 10))
  @test isapprox(r1, r2; atol=1e-12)

  # hatvalue only accepts mode 0 on the degenerate axis
  @test_throws AssertionError hatvalue(ff, (0, 1, 0))
  @test isfinite(hatvalue(ff, (0, 0, 0)))
end

@testset "degenerate axis detection helpers" begin
  lims = ((0.0,1.0), (1.234,1.234), (0.0,1.0))
  @test degenerateaxes(lims) == (false, true, false)
  @test isaxisdegenerate(lims, 2)
  @test !isaxisdegenerate(lims, 1)
end

@testset "isaxiscontinuous / detectcontinuity" begin
  fc(x) = sin(2π*x[1] + 0.3)
  fnc(x) = 0.3 + 0.5*x[1] + 0.2*sin(3*x[1])
  lims1 = ((0.0,1.0),)
  @test isaxiscontinuous(fc, lims1, 1)
  @test !isaxiscontinuous(fnc, lims1, 1)

  f2(x) = sin(2π*x[1]+0.3) * (0.3 + 0.5*x[2] + 0.2*sin(3*x[2]))
  lims2 = ((0.0,1.0),(0.0,1.0))
  @test detectcontinuity(f2, lims2) == (true, false)
end

end
