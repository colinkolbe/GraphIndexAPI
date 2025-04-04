use std::vec::Vec;
use std::marker::PhantomData;
use ndarray::{Data, Axis, ArrayBase, Array, Array1, Array2, Ix1, Ix2};

use crate::types::{Float, VFMASqEuc};


/* General definition of inner products with helper functions
 * to perform inner products on multiple vectors at once.
 * Also provides the induced distance with corresponding extensions
 * for multiple vectors. */
pub trait InnerProduct<N: Float>: Clone {
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N;
	#[inline(always)]
	fn prods
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj: &ArrayBase<D1, Ix1>, objs: &ArrayBase<D2, Ix2>) -> Array1<N> {
		objs.outer_iter()
		.map(|obj2| self.prod(obj, &obj2))
		.collect()
	}
	#[inline(always)]
	fn zip_prods
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, objs1: &ArrayBase<D1, Ix2>, objs2: &ArrayBase<D2, Ix2>) -> Array1<N> {
		objs1.outer_iter().zip(objs2.outer_iter())
		.map(|(obj1,obj2)| self.prod(&obj1, &obj2))
		.collect()
	}
	#[inline(always)]
	fn cross_prods
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, objs1: &ArrayBase<D1, Ix2>, objs2: &ArrayBase<D2, Ix2>) -> Array2<N> {
		unsafe {
			Array::from_shape_vec(
				(objs1.len_of(Axis(0)), objs2.len_of(Axis(0))),
				objs1.outer_iter()
				.flat_map(|obj1|
					objs2.outer_iter()
					.map(|obj2| self.prod(&obj1, &obj2))
					.collect::<Vec<N>>()
				)
				.collect()
			).unwrap_unchecked()
		}
	}
	#[inline(always)]
	fn self_prod
	<D1: Data<Elem=N>>
	(&self, obj: &ArrayBase<D1, Ix1>) -> N {
		self.prod(obj, obj)
	}
	#[inline(always)]
	fn self_prods
	<D1: Data<Elem=N>>
	(&self, objs: &ArrayBase<D1, Ix2>) -> Array1<N> {
		self.zip_prods(objs, objs)
	}
	
	#[inline(always)]
	fn induced_dist
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {
			PROD_COUNTER -= 1;
			DIST_COUNTER += 1;
		}
		let p = self.prod(obj1,obj2);
		let zero: N = num::Zero::zero();
		<N as num_traits::Float>::sqrt(zero.max(self.self_prod(obj1)+self.self_prod(obj2)-p-p))
		// self.self_prod(&(obj1-obj2)).sqrt()
	}
	#[inline(always)]
	fn induced_dists
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj: &ArrayBase<D1, Ix1>, objs: &ArrayBase<D2, Ix2>) -> Array1<N> {
		objs.outer_iter()
		.map(|obj2| self.induced_dist(obj, &obj2))
		.collect()
	}
	#[inline(always)]
	fn zip_induced_dists
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, objs1: &ArrayBase<D1, Ix2>, objs2: &ArrayBase<D2, Ix2>) -> Array1<N> {
		objs1.outer_iter().zip(objs2.outer_iter())
		.map(|(obj1,obj2)| self.induced_dist(&obj1, &obj2))
		.collect()
	}
	#[inline(always)]
	fn cross_induced_dists
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, objs1: &ArrayBase<D1, Ix2>, objs2: &ArrayBase<D2, Ix2>) -> Array2<N> {
		unsafe {
			Array::from_shape_vec(
				(objs1.len_of(Axis(0)), objs2.len_of(Axis(0))),
				objs1.outer_iter()
				.flat_map(|obj1|
					objs2.outer_iter()
					.map(|obj2| self.induced_dist(&obj1, &obj2))
					.collect::<Vec<N>>()
				)
				.collect()
			).unwrap_unchecked()
		}
	}
}



/* These functions are optional and solely useful for benchmarking purposes.
 * They allow to keep track of the number of distance and product computations
 * after any operations performed after the last reset.
 * To use these functions you must set the "count_operations" feature during
 * compilation with "--features count_operations". */
#[cfg(feature="count_operations")]
pub static mut PROD_COUNTER: isize = 0;
#[cfg(feature="count_operations")]
pub static mut DIST_COUNTER: isize = 0;


/* Standard dot product for real vectors. */
#[derive(Debug,Clone)]
pub struct DotProduct<N: Float> { _marker: PhantomData<N> }
impl<N: Float> DotProduct<N> {
	pub fn new() -> Self { DotProduct{_marker: PhantomData} }
}
impl<N: Float> InnerProduct<N> for DotProduct<N> {
	#[inline(always)]
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {PROD_COUNTER += 1;}
		obj1.into_iter().zip(obj2.into_iter())
		.map(|(&a,&b)| a * b)
		.reduce(|a, b| a+b)
		.unwrap_or(num::Zero::zero())
	}
}


/* Standard dot product for real vectors. */
#[derive(Debug,Clone)]
pub struct CosSim<N: Float> { _marker: PhantomData<N> }
impl<N: Float> CosSim<N> {
	#[allow(dead_code)]
	pub fn new() -> Self { CosSim{_marker: PhantomData} }
}
impl<N: Float> InnerProduct<N> for CosSim<N> {
	#[inline(always)]
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {PROD_COUNTER += 1;}
		let zero: N = num::Zero::zero();
		let sqnorm1 = obj1.into_iter().map(|&a| a*a).reduce(|a,b| a+b).unwrap_or(num::Zero::zero());
		let sqnorm2 = obj2.into_iter().map(|&a| a*a).reduce(|a,b| a+b).unwrap_or(num::Zero::zero());
		let dot = obj1.into_iter().zip(obj2.into_iter())
		.map(|(&a,&b)| a * b)
		.reduce(|a, b| a+b)
		.unwrap_or(zero);
		dot / <N as num_traits::Float>::sqrt(zero.max(sqnorm1*sqnorm2))
	}
}


/* RBF Kernel with K(x,y) = exp(-d_Euc(x,y)^2 / bandwidth) */
#[derive(Debug,Clone)]
pub struct RBFKernel<N: Float> {bandwidth: N}
impl<N: Float> RBFKernel<N> {
	#[allow(dead_code)]
	pub fn new(bandwidth: N) -> Self {
		RBFKernel{bandwidth: bandwidth}
	}
}
impl<N: Float> InnerProduct<N> for RBFKernel<N> {
	#[inline(always)]
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {PROD_COUNTER += 1;}
		let d2 = obj1.into_iter().zip(obj2.into_iter())
		.map(|(&a,&b)| a-b)
		.map(|a| a*a)
		.reduce(|a, b| a+b)
		.unwrap_or(num::Zero::zero());
		<N as num_traits::Float>::exp(-d2/self.bandwidth)
	}
}

/* Mahalanobis Kernel with <x,y> = x'C^{-1}y.
 * This induces the Mahalanobis distance for covariance C. */
#[derive(Debug,Clone)]
pub struct MahalanobisKernel<N: Float> {inv_cov: Array2<N>}
impl<N: Float> MahalanobisKernel<N> {
	#[allow(dead_code)]
	pub fn new<D: Data<Elem=N>>(inv_cov: ArrayBase<D, Ix2>) -> Self {
		MahalanobisKernel{inv_cov: inv_cov.to_owned()}
	}
}
impl<N: Float> InnerProduct<N> for MahalanobisKernel<N> {
	#[inline(always)]
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {PROD_COUNTER += 1;}
		obj1.into_iter().enumerate().zip(obj2.into_iter().enumerate())
		.map(|((i,&a),(j,&b))| a * b * unsafe { *self.inv_cov.uget([i,j]) })
		.reduce(|a, b| a+b)
		.unwrap_or(num::Zero::zero())
	}
}


/* Polynomial Kernel with <x,y> = (scale * x^Ty + bias)^degree. */
#[derive(Debug,Clone)]
pub struct PolyKernel<N: Float> {scale: N, bias: N, degree: N}
impl<N: Float> PolyKernel<N> {
	#[allow(dead_code)]
	pub fn new(scale: N, bias: N, degree: N) -> Self {
		PolyKernel{scale: scale, bias: bias, degree: degree}
	}
}
impl<N: Float> InnerProduct<N> for PolyKernel<N> {
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {PROD_COUNTER += 1;}
		let dot = obj1.into_iter().zip(obj2.into_iter())
		.map(|(&a,&b)| a * b)
		.reduce(|a, b| a+b)
		.unwrap_or(num::Zero::zero());
		<N as num_traits::Float>::powf(self.scale * dot + self.bias, self.degree)
	}
}


/* Sigmoid Kernel with <x,y> = tanh(scale * x^Ty + bias). */
#[derive(Debug,Clone)]
pub struct SigmoidKernel<N: Float> {scale: N, bias: N}
impl<N: Float> SigmoidKernel<N> {
	#[allow(dead_code)]
	pub fn new(scale: N, bias: N) -> Self {
		SigmoidKernel{scale: scale, bias: bias}
	}
}
impl<N: Float> InnerProduct<N> for SigmoidKernel<N> {
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {PROD_COUNTER += 1;}
		let dot = obj1.into_iter().zip(obj2.into_iter())
		.map(|(&a,&b)| a * b)
		.reduce(|a, b| a+b)
		.unwrap_or(num::Zero::zero());
		<N as num_traits::Float>::tanh(self.scale * dot + self.bias)
	}
}











/*
 * The part below is work in progress at best.
 * It its readily usable from within Rust but the macros to create
 * Python wrappers simply can not (yet) cope with induced inner products.
 * As for now, there are no plans to further develop this part of the
 * code, but the abstraction levels should fully empower you to build
 * your own wrappers.
 * Note: The induced inner product must be positive semi-definite!
 */


/* Abbreviated definition of a distance solely to be used with
 * the InducedInnerProduct wrapper to obtain an InnerProduct type.
 * This might be useful when working with distances that are not
 * easily translated into inner products otherwise.
 * For practical purposes use the InducedInnerProduct as it provides
 * all the extended functions for multiple vectors and uses the
 * distance functon defined here as induced distance immediately
 * at zero additional cost. */
 pub trait Distance<N: Float>: Clone {
	#[inline(always)]
	fn dist
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		self.dist_slice(&obj1.as_slice().unwrap(), &obj2.as_slice().unwrap())
	}
	fn dist_slice(&self, obj1: &&[N], obj2: &&[N]) -> N;
}
#[derive(Debug,Clone)]
pub struct InducedInnerProduct<N: Float, D: Distance<N>> {_phantom: PhantomData<N>, dist: D}
impl<N: Float, D: Distance<N>> InducedInnerProduct<N, D> {
	#[allow(dead_code)]
	pub fn new(dist: D) -> Self {
		InducedInnerProduct{_phantom: PhantomData{}, dist: dist}
	}
}
impl<N: Float, D: Distance<N>> InnerProduct<N> for InducedInnerProduct<N,D> {
	#[inline(always)]
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {
			PROD_COUNTER += 1;
		}
		let zeros: Array1<N> = Array1::zeros(obj1.shape()[0]);
		let norm1 = self.dist.dist(&zeros, obj1);
		let norm2 = self.dist.dist(&zeros, obj2);
		let dist12 = self.dist.dist(obj1, obj2);
		(norm1*norm1 + norm2*norm2 - dist12*dist12) / unsafe { N::from(2).unwrap_unchecked() }
	}
	#[inline(always)]
	fn induced_dist
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {DIST_COUNTER += 1;}
		self.dist.dist(obj1, obj2)
	}
}

#[derive(Debug,Clone)]
pub struct CosineDistance<N: Float> { _marker: PhantomData<N> }
impl<N: Float> CosineDistance<N> {
	#[allow(dead_code)]
	pub fn new() -> Self { CosineDistance{_marker: PhantomData} }
}
impl<N: Float> Distance<N> for CosineDistance<N> {
	#[inline(always)]
	fn dist_slice(&self, obj1: &&[N], obj2: &&[N]) -> N {
		#[cfg(feature="count_operations")]
		unsafe {DIST_COUNTER += 1;}
		let zero: N = num::Zero::zero();
		let one: N = num::One::one();
		let sqnorm1 = obj1.into_iter().map(|&a| a*a).reduce(|a,b| a+b).unwrap_or(num::Zero::zero());
		let sqnorm2 = obj2.into_iter().map(|&a| a*a).reduce(|a,b| a+b).unwrap_or(num::Zero::zero());
		let dot = obj1.into_iter().zip(obj2.into_iter())
		.map(|(&a,&b)| a * b)
		.reduce(|a, b| a+b)
		.unwrap_or(zero);
		let cos = dot / <N as num_traits::Float>::sqrt(zero.max(sqnorm1*sqnorm2));
		one - cos
	}

}



#[allow(unused)]
#[inline(always)]
fn simple_sq_euc<N: Float>(obj1: &[N], obj2: &[N]) -> N {
	obj1.into_iter().zip(obj2.into_iter())
	.map(|(&a,&b)| a-b)
	.map(|a| a*a)
	.reduce(|a, b| a+b)
	.unwrap_or(num::Zero::zero())
}
#[allow(unused)]
#[inline(always)]
fn optimized_sq_euc<N: Float, const LANES: usize>(v1: &[N], v2: &[N], d: usize) -> N {
	debug_assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
	debug_assert!(v1.len() == d && v2.len() == d); // bounds check
	unsafe {
		#[cfg(target_arch = "x86_64")]
		{
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const N as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const N as *const i8, _MM_HINT_T0);
		}
		let sd = d & !(LANES - 1);
		let mut vsum = [N::zero(); LANES];
		for i in (0..sd).step_by(LANES) {
			let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
			let next_i = i+LANES;
			#[cfg(target_arch = "x86_64")]
			{
				use std::arch::x86_64::*;
				_mm_prefetch(v1.get_unchecked(next_i) as *const N as *const i8, _MM_HINT_T0);
				_mm_prefetch(v2.get_unchecked(next_i) as *const N as *const i8, _MM_HINT_T0);
			}
			for j in 0..LANES {
				let x = *vv.get_unchecked(j) - *cc.get_unchecked(j);
				// emulated
				// *vsum.get_unchecked_mut(j) = x.mul_add(x, *vsum.get_unchecked(j));
				// FMA
				*vsum.get_unchecked_mut(j) += x * x;
			}
		}
		let mut sum = vsum.into_iter().sum::<N>();
		if d > sd {
			sum += (sd..d)
			.map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) })
			.map(|x| x * x)
			.sum();
		}
		sum
	}
}

#[test]
fn bench_vfma() {
	use rand::random;
	let (n_dists, dim) = (200_000, 784);
	let a: Vec<Vec<f32>> = (0..n_dists).map(|_| (0..dim).map(|_| random()).collect()).collect();
	let b: Vec<Vec<f32>> = (0..n_dists).map(|_| (0..dim).map(|_| random()).collect()).collect();
	let start_time = std::time::Instant::now();
	let dist0: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| simple_sq_euc(x.as_slice(), y.as_slice())).collect();
	let elapsed = start_time.elapsed();
	println!("simple_sq_euc: {:?}", elapsed);
	let start_time = std::time::Instant::now();
	let dist1: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| optimized_sq_euc::<f32,4>(x.as_slice(), y.as_slice(), dim)).collect();
	let elapsed = start_time.elapsed();
	println!("optimized_sq_euc: {:?}", elapsed);
	dist0.iter().cloned().zip(dist1.into_iter()).for_each(|(a,b)| assert!((a-b).abs() / a.min(b) < 1e-5, "{:?} != {:?}", a, b));
	let start_time = std::time::Instant::now();
	let dist2: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| <f32 as VFMASqEuc<4>>::sq_euc(x.as_slice(), y.as_slice(), dim)).collect();
	let elapsed = start_time.elapsed();
	println!("VFMASqEuc::sq_euc: {:?}", elapsed);
	dist0.iter().cloned().zip(dist2.into_iter()).for_each(|(a,b)| assert!((a-b).abs() / a.min(b) < 1e-5, "{:?} != {:?}", a, b));
	let start_time = std::time::Instant::now();
	let dist1: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| optimized_sq_euc::<f32,8>(x.as_slice(), y.as_slice(), dim)).collect();
	let elapsed = start_time.elapsed();
	println!("optimized_sq_euc: {:?}", elapsed);
	dist0.iter().cloned().zip(dist1.into_iter()).for_each(|(a,b)| assert!((a-b).abs() / a.min(b) < 1e-5, "{:?} != {:?}", a, b));
	let start_time = std::time::Instant::now();
	let dist2: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| <f32 as VFMASqEuc<8>>::sq_euc(x.as_slice(), y.as_slice(), dim)).collect();
	let elapsed = start_time.elapsed();
	println!("VFMASqEuc::sq_euc: {:?}", elapsed);
	dist0.iter().cloned().zip(dist2.into_iter()).for_each(|(a,b)| assert!((a-b).abs() / a.min(b) < 1e-5, "{:?} != {:?}", a, b));
}


#[derive(Debug,Clone)]
pub struct SquaredEuclideanDistance<N: Float> { _marker: PhantomData<N> }
impl<N: Float> SquaredEuclideanDistance<N> {
	#[allow(dead_code)]
	pub fn new() -> Self { SquaredEuclideanDistance{_marker: PhantomData} }
}
impl<N: Float> Distance<N> for SquaredEuclideanDistance<N> {
	#[inline(always)]
	fn dist_slice(&self, obj1: &&[N], obj2: &&[N]) -> N {
		#[cfg(feature="count_operations")]
		unsafe {DIST_COUNTER += 1;}
		// simple_sq_euc(obj1, obj2)
		#[cfg(not(target_arch = "x86_64"))]
		return optimized_sq_euc::<_,4>(obj1, obj2, obj1.len());
		#[cfg(target_arch = "x86_64")]
		return <N as VFMASqEuc<8>>::sq_euc(obj1, obj2, obj1.len());
	}
}
#[derive(Debug,Clone)]
pub struct EuclideanDistance<N: Float> { sq_euc: SquaredEuclideanDistance<N> }
impl<N: Float> EuclideanDistance<N> {
	#[allow(dead_code)]
	pub fn new() -> Self { EuclideanDistance{ sq_euc: SquaredEuclideanDistance::new() } }
}
impl<N: Float> Distance<N> for EuclideanDistance<N> {
	#[inline(always)]
	fn dist_slice(&self, obj1: &&[N], obj2: &&[N]) -> N { <N as num_traits::Float>::sqrt(self.sq_euc.dist_slice(obj1, obj2)) }
}


