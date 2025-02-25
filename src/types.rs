use ndarray_linalg::{Lapack, Scalar};
use std::iter::{Sum, Product};
use std::hash::Hash;
use std::ops::AddAssign;
use hdf5::H5Type;
use num::{NumCast, FromPrimitive, ToPrimitive, Zero, One};
use paste::paste;
use num_traits::Bounded;
use std::fmt::Debug;

#[macro_export]
macro_rules! param_struct {
	/* Matching e.g. SomeParams[Debug, Clone]<F: Float> {a: F = F::one()} */
	(
		$name:ident /* Name of the parameter struct */
		$([$($derived_type:ty),*])? /* Derived types */
		$(<$($generic_names:ident : $generic_types:path)*>)? /* Generics */
		{$($field_name:ident: $field_type:ty = $field_value:expr),*$(,)?} /* Fields */
	) => { paste::paste! {
		#[derive($($($derived_type,)*)?)]
		pub struct $name$(<$($generic_names: $generic_types),*>)? {
			$(pub $field_name: $field_type),*
		}
		impl$(<$($generic_names: $generic_types),*>)? $name$(<$($generic_names),*>)? {
			pub fn new() -> Self {
				Self {
					$($field_name: $field_value),*
				}
			}
			pub fn new_full($($field_name: Option<$field_type>,)*) -> Self {
				let mut ret = Self::new();
				$(
					if $field_name.is_some() { ret.$field_name = $field_name.unwrap(); }
				)*
				ret
			}
			$(
				pub fn [<with_ $field_name>](mut self, $field_name: $field_type) -> Self {
					self.$field_name = $field_name;
					self
				}
			)*
			$(
				pub fn [<maybe_with_ $field_name>](mut self, $field_name: Option<$field_type>) -> Self {
					if $field_name.is_some() {
						self = self.[<with_ $field_name>]($field_name.unwrap());
					}
					self
				}
			)*
		}
	}};
}
pub use param_struct;

#[macro_export]
macro_rules! trait_combiner {
	// ($combination_name: ident) => {
	// 	pub trait $combination_name {}
	// 	impl<T> $combination_name for T {}
	// };
	// ($combination_name: ident $(: $t: tt $(+ $ts: tt)*)?) => {
	// 	pub trait $combination_name $(: $t $(+ $ts)*)? {}
	// 	impl<T $(: $t $(+ $ts)*)?> $combination_name for T {}
	// };
	($combination_name: ident $([$($g: tt: $gc1: tt $(+ $gcn: tt)*),+])? $(: $t: tt $(+ $ts: tt)*)?) => {
		pub trait $combination_name$(<$($g: $gc1 $(+ $gcn)*,)+>)? $(: $t $(+ $ts)*)? {}
		impl<$($($g: $gc1 $(+ $gcn)*,)+)?T $(: $t $(+ $ts)*)?> $combination_name$(<$($g,)+>)? for T {}
	};
}
pub use trait_combiner;


#[cfg(feature="python")]
pub mod python {
	pub trait NumpyEquivalent: numpy::Element {
		fn numpy_name() -> &'static str;
	}
	macro_rules! make_numpy_equivalent {
		($(($rust_types: ty, $numpy_names: literal)),*) => {
			$(make_numpy_equivalent!($rust_types, $numpy_names);)*
		};
		($rust_type: ty, $numpy_name: literal) => {
			impl NumpyEquivalent for $rust_type {
				fn numpy_name() -> &'static str {
					$numpy_name
				}
			}
		};
	}
	use half::f16;
	make_numpy_equivalent!(
		(f16, "float16"), (f32, "float32"), (f64, "float64"), /*(f128, "float128"), // Not yet supported it appears */
		(bool, "bool_"),
		(u8, "uint8"), (u16, "uint16"),	(u32, "uint32"), (u64, "uint64"),
		(i8, "int8"), (i16, "int16"),	(i32, "int32"), (i64, "int64")
	);
}

trait_combiner!(Static: 'static);
trait_combiner!(Sync: (std::marker::Send)+(std::marker::Sync));
#[cfg(feature="python")]
trait_combiner!(Number: (python::NumpyEquivalent)+Bounded+H5Type+NumCast+FromPrimitive+ToPrimitive+Zero+One+Sum+Product+AddAssign+Copy+Clone+Debug);
#[cfg(not(feature="python"))]
trait_combiner!(Number: Bounded+H5Type+NumCast+FromPrimitive+ToPrimitive+Zero+One+Sum+Product+AddAssign+Copy+Clone+Debug);

macro_rules! make_num_variants {
	($($baseType:ident),*) => {
		paste! {
			$(
				trait_combiner!([<Static $baseType>]: $baseType+Static);
				trait_combiner!([<Sync $baseType>]: $baseType+Sync);
				trait_combiner!([<StaticSync $baseType>]: [<Sync $baseType>]+Static);
			)*
		}
	};
}

trait_combiner!(Integer: Number+(num::Integer));
trait_combiner!(UnsignedInteger: Hash+Integer+(num::Unsigned));
trait_combiner!(SignedInteger: Integer+(num::Signed));
trait_combiner!(Float: Scalar+Lapack+Number+(num::Float));

make_num_variants!(Number, Integer, UnsignedInteger, SignedInteger, Float);

