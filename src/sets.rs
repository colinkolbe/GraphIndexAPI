use foldhash::{HashMap, HashSet};
use std::sync::{Arc,Mutex, RwLock};

use crate::{bit_vectors::{BitVector,BitVectorMut}, types::{SyncUnsignedInteger, UnsignedInteger}};

#[inline(always)]
pub fn zero_vec<T>(v: &mut Vec<T>) {
	unsafe{
		std::ptr::write_bytes(v.get_unchecked_mut(0) as *mut T, 0, v.capacity());
	}
}
#[inline(always)]
pub fn zero_init_vec<T>(size: usize) -> Vec<T> {
	unsafe{
		let mut v = Vec::with_capacity(size);
		v.set_len(size);
		zero_vec(&mut v);
		v
	}
}


/* 
 * Thread-safe buffer manager implementation.
 * For each datatype and buffer size, a manager is automatically
 * created and managed by a central CEO. The CEO is a singleton
 * per datatype that manages all managers and cleans up unused managers.
 * Each manager holds a pool of clean arrays and can be used to
 * allocate and deallocate arrays of the specified size.
 * The manager is automatically cleaned up once all external references
 * to it are dropped.
 */
struct BufferCEO<R: SyncUnsignedInteger> {
	managers: HashMap<usize, ManagerArc<R>>,
}
impl<R: SyncUnsignedInteger> BufferCEO<R> {
	#[inline(always)]
	pub fn new() -> Self {
		Self {
			managers: HashMap::default(),
		}
	}
	#[inline(always)]
	pub fn get_manager(&mut self, array_size: usize) -> ManagerArc<R> {
		let manager = self.managers.get(&array_size);
		let mut manager = if manager.is_some() {
			ManagerArc::clone(manager.unwrap())
		} else {
			let manager = ManagerArc::new(BufferManager::new(array_size), false);
			self.managers.insert(array_size, ManagerArc::clone(&manager));
			manager
		};
		/* Flag manager as owned by set to invoke cleaning once this reference dies */
		manager.invokes_cleaning = true;
		manager
	}
	#[inline(always)]
	pub fn drop_manager(&mut self, array_size: usize) {
		self.managers.remove(&array_size);
	}
}
#[derive(Clone)]
pub struct ManagerArc<R: SyncUnsignedInteger> {
	arc: Arc<BufferManager<R>>,
	invokes_cleaning: bool,
}
impl<R: SyncUnsignedInteger> ManagerArc<R> {
	#[inline(always)]
	fn new(manager: BufferManager<R>, owned_by_set: bool) -> Self {
		Self { arc: Arc::new(manager), invokes_cleaning: owned_by_set }
	}
}
impl<R: SyncUnsignedInteger> std::ops::Deref for ManagerArc<R> {
	type Target = BufferManager<R>;
	#[inline(always)]
	fn deref(&self) -> &Self::Target {
		&self.arc
	}
}
impl<R: SyncUnsignedInteger> Drop for ManagerArc<R> {
	#[inline(always)]
	fn drop(&mut self) {
		/* Test if this is the last non-CEO arc to the manager and if so drop the whole manager */
		if self.invokes_cleaning && Arc::strong_count(&self.arc) == 2 {
			let ceo = get_buffer_ceo::<R>();
			ceo.lock().unwrap().drop_manager(self.arc.array_size);
		}
	}
}
pub struct BufferManager<R: SyncUnsignedInteger> {
	clean_arrays: Arc<Mutex<Vec<Vec<R>>>>,
	array_size: usize,
}
impl<R: SyncUnsignedInteger> BufferManager<R> {
	#[inline(always)]
	pub fn new(array_size: usize) -> Self {
		Self {
			clean_arrays: Arc::new(Mutex::new(Vec::new())),
			array_size,
		}
	}
	#[inline(always)]
	pub fn prepare_clean(&self) {
		let clean_array_arc = Arc::clone(&self.clean_arrays);
		let array_size = self.array_size;
		rayon::spawn(move || {
			let vec = zero_init_vec(array_size);
			clean_array_arc.lock().unwrap().push(vec);
		});
	}
	#[inline(always)]
	pub fn get_clean_array(&self) -> Vec<R> {
		let mut clean_arrays = self.clean_arrays.lock().unwrap();
		if clean_arrays.is_empty() {
			zero_init_vec(self.array_size)
		} else {
			clean_arrays.pop().unwrap()
		}
	}
	#[inline(always)]
	pub fn return_dirty_array(&self, mut array: Vec<R>) {
		let clean_array_arc = Arc::clone(&self.clean_arrays);
		rayon::spawn(move || {
			zero_vec(&mut array);
			clean_array_arc.lock().unwrap().push(array);
		});
	}
	#[inline(always)]
	pub fn swap_with_clean_array(&self, array: &mut Vec<R>) {
		let mut clean_array = self.get_clean_array();
		std::mem::swap(&mut clean_array, array);
		/* `clean_array` now contains the dirty returned array, so let's clean it! */
		self.return_dirty_array(clean_array);
	}
}
/* Accessor functions */
#[inline(always)]
fn get_buffer_ceo<R: SyncUnsignedInteger>() -> Arc<Mutex<BufferCEO<R>>> {
	generic_singleton::get_or_init!(|| RwLock::new(Arc::new(Mutex::new(BufferCEO::new())))).write().unwrap().clone()
}
#[inline(always)]
/// Returns an arc for a specific array size.
/// Request and hold this arc for as long as you want the manager to live.
/// As soon as all external arcs to the manager go out of scope, the manager will be dropped.
pub fn get_manager_arc<R: SyncUnsignedInteger>(array_size: usize) -> ManagerArc<R> {
	let ceo = get_buffer_ceo::<R>();
	let manager = ceo.lock().unwrap().get_manager(array_size);
	manager
}


pub trait HashSetLike<T> {
	fn new(capacity: usize) -> Self;
	fn insert(&mut self, value: T) -> bool;
	fn contains(&self, value: &T) -> bool;
	fn clear(&mut self);
	#[inline(always)]
	fn reserve(&mut self, _capacity: usize) {}
}

impl<T: Eq+std::hash::Hash> HashSetLike<T> for HashSet<T> {
	#[inline(always)]
	fn new(_: usize) -> Self {
		HashSet::default()
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		self.insert(value)
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		self.contains(value)
	}
	#[inline(always)]
	fn clear(&mut self) {
		self.clear()
	}
	#[inline(always)]
	fn reserve(&mut self, capacity: usize) {
		self.reserve(capacity);
	}
}

pub struct ApproxSet<T: UnsignedInteger, R: SyncUnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<R>,
	set_counter: R,
	perm: crate::random::RandomPermutationGenerator,
}
impl<T: UnsignedInteger, R: SyncUnsignedInteger> ApproxSet<T, R> {
	#[inline(always)]
	fn new(capacity: usize, max_size: Option<usize>, n_rounds: Option<usize>) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		Self {
			_phantom: std::marker::PhantomData,
			current: zero_init_vec(max_size.unwrap_or(capacity/100)),
			set_counter: R::one(),
			perm: crate::random::RandomPermutationGenerator::new(capacity, n_rounds.unwrap_or(3)),
		}
	}
}
impl<T: UnsignedInteger, R: SyncUnsignedInteger> HashSetLike<T> for ApproxSet<T, R> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		Self::new(capacity, None, None)
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let value = self.perm.apply_rounds(value.to_usize().unwrap_unchecked()) % self.current.len();
			let cur = self.current.get_unchecked_mut(value);
			if *cur == self.set_counter {
				false
			} else {
				*cur = self.set_counter;
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe {
			let value = self.perm.apply_rounds(value.to_usize().unwrap_unchecked()) % self.current.len();
			*self.current.get_unchecked(value) == self.set_counter
		}
	}
	#[inline(always)]
	fn clear(&mut self) {
		/* Incrementing with wrapping to jump back to 0 after max iterations */
		self.set_counter = self.set_counter.wrapping_add(&R::one());
		/* If wrapping, the current array is invalidated and must be zeroed again */
		if self.set_counter == R::zero() {
			zero_vec(&mut self.current);
			self.perm.reset_rand_state();
			self.set_counter = R::one();
		}
	}
}

pub struct ApproxBitSet<T: UnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<u64>,
	perm: crate::random::RandomPermutationGenerator,
}
impl<T: UnsignedInteger> ApproxBitSet<T> {
	#[inline(always)]
	fn new(capacity: usize, max_size: Option<usize>, n_rounds: Option<usize>) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		let max_size = (max_size.unwrap_or(capacity / 100)+63) / 64;
		Self {
			_phantom: std::marker::PhantomData,
			current: zero_init_vec(max_size),
			perm: crate::random::RandomPermutationGenerator::new(capacity, n_rounds.unwrap_or(3)),
		}
	}
}
impl<T: UnsignedInteger> HashSetLike<T> for ApproxBitSet<T> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		Self::new(capacity, None, None)
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let value = self.perm.apply_rounds(value.to_usize().unwrap_unchecked()) % (self.current.len()*64);
			let cur = self.current.get_bit_unchecked(value);
			if cur {
				false
			} else {
				self.current.set_bit_unchecked(value, true);
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe {
			let value = self.perm.apply_rounds(value.to_usize().unwrap_unchecked()) % (self.current.len()*64);
			self.current.get_bit_unchecked(value)
		}
	}
	#[inline(always)]
	fn clear(&mut self) {
		zero_vec(&mut self.current);
		self.perm.reset_rand_state();
	}
}

pub struct RoaringSet<T: UnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	roaring: roaring::RoaringBitmap,
}
impl<T: UnsignedInteger> HashSetLike<T> for RoaringSet<T> {
	#[inline(always)]
	fn new(_: usize) -> Self {
		Self {
			_phantom: std::marker::PhantomData,
			roaring: roaring::RoaringBitmap::new()
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe { self.roaring.insert(value.to_u32().unwrap_unchecked()) }
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { self.roaring.contains(value.to_u32().unwrap_unchecked()) }
	}
	#[inline(always)]
	fn clear(&mut self) {
		self.roaring.clear()
	}
}
pub struct RoaringTreeSet<T: UnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	roaring: roaring::RoaringTreemap,
}
impl<T: UnsignedInteger> HashSetLike<T> for RoaringTreeSet<T> {
	#[inline(always)]
	fn new(_: usize) -> Self {
		Self {
			_phantom: std::marker::PhantomData,
			roaring: roaring::RoaringTreemap::new()
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe { self.roaring.insert(value.to_u64().unwrap_unchecked()) }
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { self.roaring.contains(value.to_u64().unwrap_unchecked()) }
	}
	#[inline(always)]
	fn clear(&mut self) {
		self.roaring.clear()
	}
}

pub struct BitSet<T: UnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<u64>,
}
impl<T: UnsignedInteger> HashSetLike<T> for BitSet<T> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		let capacity = (capacity + 63) / 64;
		Self {
			_phantom: std::marker::PhantomData,
			current: zero_init_vec(capacity),
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let cur = self.current.get_bit_unchecked(value.to_usize().unwrap_unchecked());
			if cur {
				false
			} else {
				self.current.set_bit_unchecked(value.to_usize().unwrap_unchecked(), true);
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { self.current.get_bit_unchecked(value.to_usize().unwrap_unchecked()) }
	}
	#[inline(always)]
	fn clear(&mut self) {
		zero_vec(&mut self.current);
	}
}

pub struct TrackingBitSet<T: UnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<u64>,
	set_bits: Vec<usize>,
}
impl<T: UnsignedInteger> HashSetLike<T> for TrackingBitSet<T> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		let capacity = (capacity + 63) / 64;
		Self {
			_phantom: std::marker::PhantomData,
			current: zero_init_vec(capacity),
			set_bits: Vec::new(),
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let value = value.to_usize().unwrap_unchecked();
			let cur = self.current.get_bit_unchecked(value);
			if cur {
				false
			} else {
				self.current.set_bit_unchecked(value, true);
				self.set_bits.push(value);
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { self.current.get_bit_unchecked(value.to_usize().unwrap_unchecked()) }
	}
	#[inline(always)]
	fn clear(&mut self) {
		unsafe {
			self.set_bits.iter().for_each(|&bit| *self.current.get_unchecked_mut(bit/64) = 0);
			self.set_bits.clear();
		}
	}
}

pub struct SwappingBitSet<T: UnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<u64>,
	next: Arc<Mutex<Vec<u64>>>,
}
impl<T: UnsignedInteger> HashSetLike<T> for SwappingBitSet<T> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		let capacity = (capacity + 63) / 64;
		Self {
			_phantom: std::marker::PhantomData,
			current: zero_init_vec(capacity),
			next: Arc::new(Mutex::new(zero_init_vec(capacity))),
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let cur = self.current.get_bit_unchecked(value.to_usize().unwrap_unchecked());
			if cur {
				false
			} else {
				self.current.set_bit_unchecked(value.to_usize().unwrap_unchecked(), true);
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { self.current.get_bit_unchecked(value.to_usize().unwrap_unchecked()) }
	}
	#[inline(always)]
	fn clear(&mut self) {
		/* Swapping with the backup array and zeroing the invalidated array */
		std::mem::swap(&mut self.current, self.next.lock().unwrap().as_mut());
		let next_arc = Arc::clone(&self.next);
		rayon::spawn(move || zero_vec(next_arc.lock().unwrap().as_mut()));
	}
}

pub struct ManagedBitSet<T: UnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	manager: ManagerArc<u64>,
	array: Vec<u64>,
}
impl<T: UnsignedInteger> HashSetLike<T> for ManagedBitSet<T> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		let capacity = (capacity + 63) / 64;
		let ceo = get_buffer_ceo::<u64>();
		let manager = ceo.lock().unwrap().get_manager(capacity);
		let array = manager.get_clean_array();
		let result = Self {
			_phantom: std::marker::PhantomData,
			manager,
			array,
		};
		result
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let cur = self.array.get_bit_unchecked(value.to_usize().unwrap_unchecked());
			if cur {
				false
			} else {
				self.array.set_bit_unchecked(value.to_usize().unwrap_unchecked(), true);
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { self.array.get_bit_unchecked(value.to_usize().unwrap_unchecked()) }
	}
	#[inline(always)]
	fn clear(&mut self) {
		self.manager.swap_with_clean_array(&mut self.array);
	}
}


pub struct ReallocSet<T: UnsignedInteger, R: SyncUnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<R>,
	set_counter: R,
}
impl<T: UnsignedInteger, R: SyncUnsignedInteger> HashSetLike<T> for ReallocSet<T, R> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		Self {
			_phantom: std::marker::PhantomData,
			current: vec![R::zero(); capacity],
			set_counter: R::one(),
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let cur = self.current.get_unchecked_mut(value.to_usize().unwrap_unchecked());
			if *cur == self.set_counter {
				false
			} else {
				*cur = self.set_counter;
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { *self.current.get_unchecked(value.to_usize().unwrap_unchecked()) == self.set_counter }
	}
	#[inline(always)]
	fn clear(&mut self) {
		/* Incrementing with wrapping to jump back to 0 after max iterations */
		self.set_counter = self.set_counter.wrapping_add(&R::one());
		/* If wrapping, the current array is invalidated and must be zeroed again */
		if self.set_counter == R::zero() {
			self.current = vec![R::zero(); self.current.len()];
			self.set_counter = R::one();
		}
	}
}


pub struct NAryArraySet<T: UnsignedInteger, R: SyncUnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<R>,
	set_counter: R,
	arity: usize,
	counter_mask: R,
}
impl<T: UnsignedInteger, R: SyncUnsignedInteger+std::ops::BitAnd<Output=R>+std::ops::BitOr<Output=R>> HashSetLike<T> for NAryArraySet<T, R> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		let arity = 4;
		let capacity = (capacity+arity-1) / arity;
		Self {
			_phantom: std::marker::PhantomData,
			current: zero_init_vec(capacity),
			set_counter: R::zero(),
			arity: arity,
			counter_mask: R::from((!((1<<arity)-1)) & R::max_value().to_usize().unwrap()).unwrap(),
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let value = value.to_usize().unwrap_unchecked();
			/* Position of this items bit in the array */
			let array_offset = value / self.arity;
			/* Selecting the bit corresponding to this item */
			let value_arity_mask = R::from(1 << (value % self.arity)).unwrap();
			/* Object containing this items bit */
			let cur = self.current.get_unchecked_mut(array_offset);
			/* Test whether the counter part of the current value matches */
			if (*cur & self.counter_mask) == self.set_counter {
				/* Test whether the value part of the current value matches */
				if (*cur & value_arity_mask) != R::zero() {
					/* Item is already included in the current set representation */
					false
				} else {
					/* Some item of this array object is already included, just add the bit */
					*cur = *cur | value_arity_mask;
					true
				}
			} else {
				/* No item of this array object is included, set all item bits to 0 and update the counter part */
				*cur = self.set_counter | value_arity_mask;
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { 
			let value = value.to_usize().unwrap_unchecked();
			let array_offset = value / self.arity;
			let value_arity_mask = R::from(1 << (value % self.arity)).unwrap();
			let cur = self.current.get_unchecked(array_offset);
			((*cur & self.counter_mask) == self.set_counter) && ((*cur & value_arity_mask) != R::zero())
		}
	}
	#[inline(always)]
	fn clear(&mut self) {
		unsafe {
			/* Incrementing with wrapping to jump back to 0 after max iterations */
			self.set_counter = self.set_counter.wrapping_add(&R::from(1 << self.arity).unwrap_unchecked());
			/* If wrapping, the current array is invalidated and must be zeroed again */
			if self.set_counter == R::zero() {
				zero_vec(&mut self.current);
			}
		}
	}
}

pub struct ArraySet<T: UnsignedInteger, R: SyncUnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<R>,
	set_counter: R,
}
impl<T: UnsignedInteger, R: SyncUnsignedInteger> HashSetLike<T> for ArraySet<T, R> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		Self {
			_phantom: std::marker::PhantomData,
			current: zero_init_vec(capacity),
			set_counter: R::one(),
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let cur = self.current.get_unchecked_mut(value.to_usize().unwrap_unchecked());
			if *cur == self.set_counter {
				false
			} else {
				*cur = self.set_counter;
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { *self.current.get_unchecked(value.to_usize().unwrap_unchecked()) == self.set_counter }
	}
	#[inline(always)]
	fn clear(&mut self) {
		/* Incrementing with wrapping to jump back to 0 after max iterations */
		self.set_counter = self.set_counter.wrapping_add(&R::one());
		/* If wrapping, the current array is invalidated and must be zeroed again */
		if self.set_counter == R::zero() {
			zero_vec(&mut self.current);
			self.set_counter = R::one();
		}
	}
}

pub struct SwappingArraySet<T: UnsignedInteger, R: SyncUnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	current: Vec<R>,
	next: Arc<Mutex<Vec<R>>>,
	set_counter: R,
}
impl<T: UnsignedInteger, R: SyncUnsignedInteger> HashSetLike<T> for SwappingArraySet<T, R> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		Self {
			_phantom: std::marker::PhantomData,
			current: zero_init_vec(capacity),
			next: Arc::new(Mutex::new(zero_init_vec(capacity))),
			set_counter: R::one(),
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let cur = self.current.get_unchecked_mut(value.to_usize().unwrap_unchecked());
			if *cur == self.set_counter {
				false
			} else {
				*cur = self.set_counter;
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { *self.current.get_unchecked(value.to_usize().unwrap_unchecked()) == self.set_counter }
	}
	#[inline(always)]
	fn clear(&mut self) {
		/* Incrementing with wrapping to jump back to 0 after max iterations */
		self.set_counter = self.set_counter.wrapping_add(&R::one());
		/* If wrapping, the current array is invalidated and must be zeroed again */
		/* Swapping with the backup array and zeroing the invalidated array */
		if self.set_counter == R::zero() {
			std::mem::swap(&mut self.current, self.next.lock().unwrap().as_mut());
			self.set_counter = R::one();
			let next_arc = Arc::clone(&self.next);
			rayon::spawn(move || zero_vec(next_arc.lock().unwrap().as_mut()));
		}
	}
}


pub struct ManagedArraySet<T: UnsignedInteger, R: SyncUnsignedInteger> {
	_phantom: std::marker::PhantomData<T>,
	manager: ManagerArc<R>,
	array: Vec<R>,
	set_counter: R,
}
impl<T: UnsignedInteger, R: SyncUnsignedInteger> HashSetLike<T> for ManagedArraySet<T, R> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		assert!(capacity < T::max_value().to_usize().unwrap());
		let ceo = get_buffer_ceo::<R>();
		let manager = ceo.lock().unwrap().get_manager(capacity);
		let array = manager.get_clean_array();
		let result = Self {
			_phantom: std::marker::PhantomData,
			manager,
			array,
			set_counter: R::one(),
		};
		result
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		unsafe {
			let cur = self.array.get_unchecked_mut(value.to_usize().unwrap_unchecked());
			if *cur == self.set_counter {
				false
			} else {
				*cur = self.set_counter;
				true
			}
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		unsafe { *self.array.get_unchecked(value.to_usize().unwrap_unchecked()) == self.set_counter }
	}
	#[inline(always)]
	fn clear(&mut self) {
		/* Incrementing with wrapping to jump back to 0 after max iterations */
		self.set_counter = self.set_counter.wrapping_add(&R::one());
		/* If wrapping, the current array is invalidated and must be zeroed again */
		/* Swapping with the backup array and zeroing the invalidated array */
		if self.set_counter == R::zero() {
			self.manager.swap_with_clean_array(&mut self.array);
			self.set_counter = R::one();
		}
	}
}


#[cfg(test)]
macro_rules! array_set_test {
	(correct $val_type: ident, [$($type: ident <$g1: ident $(, $g2: ident)?>),+$(,)?], $n_threads: ident, $n_clears: ident, $n_inserts: ident, $max_val: ident) => {
		paste::paste!{
			let randint = |max| (random::<$val_type>() as usize % max) as $val_type;
			let mut hash_set = HashSet::<$val_type>::new($max_val);
			$(
				#[allow(non_snake_case)]
				let mut [<$type _set_ $g1 $(_ $g2)?>] = <$type::<$g1$(,$g2)?> as HashSetLike<$g1>>::new($max_val);
			)+
			/* Verify correctness of all sets */
			(0..$n_clears).for_each(|_| {
				hash_set.clear();
				paste::paste!{ $([<$type _set_ $g1 $(_ $g2)?>].clear();)+ };
				(0..$n_inserts).for_each(|_| {
					let val = randint($max_val);
					let truth = hash_set.insert(val);
					$(assert_eq!([<$type _set_ $g1 $(_ $g2)?>].insert(val), truth, "Bad insertion for set type {}", stringify!($type));)+
				});
				(0..$max_val).for_each(|val| {
					let truth = hash_set.contains(&(val as T));
					$(assert_eq!([<$type _set_ $g1 $(_ $g2)?>].contains(&(val as T)), truth, "Bad contains result for set type {}", stringify!($type));)+
				});
			});
		};
	};
	(bench [$($type: ident <$g1: ident $(, $g2: ident)?>),+$(,)?], $n_threads: ident, $n_clears: ident, $n_inserts: ident, $max_val: ident) => {
		array_set_test!(bench clear [$($type<$g1$(,$g2)?>,)+], $n_threads, $n_clears, $max_val);
		array_set_test!(bench insert [$($type<$g1$(,$g2)?>,)+], $n_threads, $n_clears, $n_inserts, $max_val);
	};
	(bench clear [$($type: ident <$g1: ident $(, $g2: ident)?>),+$(,)?], $n_threads: ident, $n_clears: ident, $max_val: ident) => {
		$(
			let start_time = std::time::Instant::now();
			(0..$n_threads).into_par_iter().for_each(|_| {
				let mut set = <$type::<$g1$(,$g2)?> as HashSetLike<$g1>>::new($max_val);
				(0..$n_clears).for_each(|_| {
					set.clear();
					std::hint::black_box(&set);
				});
			});
			let elapsed_time = start_time.elapsed();
			println!("Clearing {}: {:?}", stringify!($type), elapsed_time);
		)+
	};
	(bench insert [$($type: ident <$g1: ident $(, $g2: ident)?>),+$(,)?], $n_threads: ident, $n_clears: ident, $n_inserts: ident, $max_val: ident) => {
		$(
			let randint = |max| (random::<$g1>() as usize % max) as $g1;
			let start_time = std::time::Instant::now();
			(0..$n_threads).into_par_iter().for_each(|_| {
				let mut set = <$type::<$g1$(,$g2)?> as HashSetLike<$g1>>::new($max_val);
				(0..$n_clears).for_each(|_| {
					set.clear();
					(0..$n_inserts).for_each(|_| {
						let val = randint($max_val);
						set.insert(val);
					});
					std::hint::black_box(&set);
				});
			});
			let elapsed_time = start_time.elapsed();
			println!("Inserting {}: {:?}", stringify!($type), elapsed_time);
		)+
	};
}

#[test]
fn test_array_sets() {
	use rand::random;
	type R = u8;
	type T = usize;
	/* Correctness Parameters */
	let n_clears = 1_000;
	let n_inserts = 2_000;
	let max_val = 100_000;
	array_set_test!(correct T, [
		RoaringSet<T>,
		RoaringTreeSet<T>,
		BitSet<T>,
		SwappingBitSet<T>,
		ReallocSet<T,R>,
		NAryArraySet<T,R>,
		ArraySet<T,R>,
		SwappingArraySet<T,R>,
		ManagedArraySet<T,R>,
	], n_threads, n_clears, n_inserts, max_val);

	/* Measure error rate in approx set */
	let randint = |max| (random::<T>() as usize % max) as T;
	let mut hash_set = HashSet::<T>::new(max_val);
	let mut approx_set = ApproxSet::<T,R>::new(max_val, None, None);
	let mut approx_bit_set = ApproxBitSet::<T>::new(max_val, None, None);
	let mut insert_errors = 0usize;
	let mut contains_errors = 0usize;
	let mut bit_insert_errors = 0usize;
	let mut bit_contains_errors = 0usize;
	(0..n_clears).for_each(|_| {
		hash_set.clear();
		approx_set.clear();
		approx_bit_set.clear();
		(0..n_inserts).for_each(|_| {
			let val = randint(max_val);
			let truth = hash_set.insert(val);
			if approx_set.insert(val) != truth { insert_errors += 1; }
			if approx_bit_set.insert(val) != truth { bit_insert_errors += 1; }
		});
		(0..max_val).for_each(|val| {
			let truth = hash_set.contains(&(val as T));
			if approx_set.contains(&(val as T)) != truth { contains_errors += 1; }
			if approx_bit_set.contains(&(val as T)) != truth { bit_contains_errors += 1; }
		});
	});
	println!("Insert Error rate: {}/{} = {:.2}%", insert_errors, n_clears*n_inserts, insert_errors as f64 / (n_clears*n_inserts) as f64 * 100.0);
	println!("Contains Error rate: {}/{} = {:.2}%", contains_errors, n_clears*max_val, contains_errors as f64 / (n_clears*max_val) as f64 * 100.0);
	println!("Bit Insert Error rate: {}/{} = {:.2}%", bit_insert_errors, n_clears*n_inserts, bit_insert_errors as f64 / (n_clears*n_inserts) as f64 * 100.0);
	println!("Bit Contains Error rate: {}/{} = {:.2}%", bit_contains_errors, n_clears*max_val, bit_contains_errors as f64 / (n_clears*max_val) as f64 * 100.0);

	/* Benchmark Parameters */
	use rayon::iter::{ParallelIterator,IntoParallelIterator};
	let n_threads = 16;
	let max_val = 6_000_000;
	let degree = 50usize;
	let ef = 100;
	let n_clears = max_val / n_threads * ((max_val as f64).log2() / (degree as f64).log2()) as usize / 100;
	let n_inserts = degree * ef; /* Lowest level typical load */
	// let n_inserts = degree * 3; /* Higher level typical load */

	let manager_arc = get_manager_arc::<R>(max_val);
	(0..n_threads+1+n_threads/5).for_each(|_| manager_arc.prepare_clean());
	
	array_set_test!(bench insert [
		ApproxSet<T,R>,
		ApproxBitSet<T>,
		RoaringSet<T>,
		RoaringTreeSet<T>,
		HashSet<T>,
		BitSet<T>,
		SwappingBitSet<T>,
		ReallocSet<T,R>,
		NAryArraySet<T,R>,
		ArraySet<T,R>,
		SwappingArraySet<T,R>,
		ManagedArraySet<T,R>,
	], n_threads, n_clears, n_inserts, max_val);
}



/* Wrapping the used sets in an enum allows to create a vec
 * over an arbitrary combination of those */
pub enum HashOrBitset<T: SyncUnsignedInteger+std::hash::Hash> {
	Hash(foldhash::HashSet<T>),
	Bit(BitSet<T>),
}
impl<T: SyncUnsignedInteger+std::hash::Hash> HashOrBitset<T> {
	#[inline(always)]
	pub fn new_bit(capacity: usize) -> Self {
		HashOrBitset::Bit(<BitSet<T> as HashSetLike<T>>::new(capacity))
	}
	#[inline(always)]
	pub fn new_hash(capacity: usize) -> Self {
		HashOrBitset::Hash(<foldhash::HashSet<T> as HashSetLike<T>>::new(capacity))
	}
}
impl<T: SyncUnsignedInteger+std::hash::Hash> HashSetLike<T> for HashOrBitset<T> {
	#[inline(always)]
	fn new(capacity: usize) -> Self {
		/* 5M is approximately the empirically dervied threshold
		 * when hashsets become faster in a HNSW-typical setting */
		if capacity <= 5_000_000 {
			HashOrBitset::new_bit(capacity)
		} else {
			HashOrBitset::new_hash(capacity)
		}
	}
	#[inline(always)]
	fn reserve(&mut self, additional: usize) {
		match self {
			HashOrBitset::Hash(h) => h.reserve(additional),
			HashOrBitset::Bit(b) => b.reserve(additional),
		}
	}
	#[inline(always)]
	fn insert(&mut self, value: T) -> bool {
		match self {
			HashOrBitset::Hash(h) => h.insert(value),
			HashOrBitset::Bit(b) => b.insert(value),
		}
	}
	#[inline(always)]
	fn contains(&self, value: &T) -> bool {
		match self {
			HashOrBitset::Hash(h) => h.contains(value),
			HashOrBitset::Bit(b) => b.contains(value),
		}
	}
	#[inline(always)]
	fn clear(&mut self) {
		match self {
			HashOrBitset::Hash(h) => h.clear(),
			HashOrBitset::Bit(b) => b.clear(),
		}
	}
}
#[test]
fn test_hash_or_bitset() {
	let mut vec: Vec<HashOrBitset<usize>> = Vec::new();
	vec.push(HashOrBitset::new_bit(10));
	vec.push(HashOrBitset::new_bit(1_000));
	vec.push(HashOrBitset::new_bit(100_000));
	vec.push(HashOrBitset::new_hash(10_000_000));
	std::hint::black_box(&vec);
}





pub mod zero_benching {
	#[inline(always)]
	pub fn zero_initialized_vec_init<T: num::Zero + Copy>(size: usize) -> Vec<T> {
		vec![T::zero(); size]
	}
	#[inline(always)]
	pub fn zero_initialized_memset<T: num::Zero + Copy + Sized>(size: usize) -> Vec<T> {
		let mut v = Vec::with_capacity(size);
		unsafe{
			v.set_len(size);
			std::ptr::write_bytes(v.get_unchecked_mut(0) as *mut T, 0, size);
		}
		v
	}
	#[inline(always)]
	pub fn zero_initialized_memset_chunked<T: num::Zero + Copy + Sized>(size: usize) -> Vec<T> {
		const CHUNK_SIZE: usize = 64;
		let mut v = Vec::with_capacity(size);
		unsafe{
			v.set_len(size);
			(0..size).step_by(CHUNK_SIZE).for_each(|offset| {
				let eff_chunk_size = CHUNK_SIZE.min(size-offset);
				std::ptr::write_bytes(v.get_unchecked_mut(offset) as *mut T, 0, eff_chunk_size);
			});
		}
		v
	}
	#[inline(always)]
	pub fn zero_initialized_avx<T: num::Zero + Copy + Sized>(size: usize) -> Vec<T> {
		let bytesize = std::mem::size_of::<T>();
		let total_bytes = bytesize * size;
		let mut v = Vec::with_capacity(size);
		unsafe{
			v.set_len(size);
			use std::arch::x86_64::*;
			let start = v.get_unchecked_mut(0) as *mut T as *mut u8;
			let zeros = _mm256_setzero_si256();
			if start as usize % 32 != 0 {
				_mm256_storeu_si256(start as *mut __m256i, zeros);
				_mm256_storeu_si256((start as usize+total_bytes-32) as *mut __m256i, zeros);
			}
			let start = ((start as usize+31) & !31) as *mut u8;
			(0..total_bytes-32).step_by(32).for_each(|offset| {
				_mm_prefetch((start as usize+offset+32) as *const i8, _MM_HINT_T0);
				_mm256_store_si256((start as usize+offset) as *mut __m256i, zeros);
			});
		}
		v
	}
	#[inline(always)]
	pub fn zero_initialized_fill<T: num::Zero + Copy + Sized>(size: usize) -> Vec<T> {
		let mut v = Vec::with_capacity(size);
		unsafe{
			v.set_len(size);
			v.fill(T::zero());
		}
		v
	}

	#[inline(never)]
	pub fn zero_initialized_vec_init_u16(size: usize) -> Vec<u16> { zero_initialized_vec_init(size) }
	#[inline(never)]
	pub fn zero_initialized_memset_u16(size: usize) -> Vec<u16> { zero_initialized_memset(size) }
	#[inline(never)]
	pub fn zero_initialized_memset_chunked_u16(size: usize) -> Vec<u16> { zero_initialized_memset(size) }
	#[inline(never)]
	pub fn zero_initialized_avx_u16(size: usize) -> Vec<u16> { zero_initialized_avx(size) }
	#[inline(never)]
	pub fn zero_initialized_fill_u16(size: usize) -> Vec<u16> { zero_initialized_fill(size) }


	#[allow(unused)]
	fn bench<A,B,F: FnMut(A) -> (B)>(name: &str, mut f: F, args: A) -> B {
		let start_time = std::time::Instant::now();
		let result = f(args);
		let elapsed_time = start_time.elapsed();
		println!("{}: {:?}", name, elapsed_time);
		std::hint::black_box(&result);
		result
	}
	#[allow(unused)]
	fn bench_it<A: Clone,B,F: FnMut(A) -> (B)>(name: &str, mut f: F, args: A, iters: usize) -> B {
		assert!(iters > 0);
		bench(name, |args| {
			let a = (0..iters).map(|_| f(args.clone())).reduce(|a,b| b).unwrap();
			std::hint::black_box(&a);
			a
		}, args)
	}
	macro_rules! _bench_for_type {
		(call $name: literal, $fun: ident, $type: ident, $size: literal) => {
			bench($name, |size| assert!($fun::<$type>(size).iter().all(|&x| x == 0 as $type)), $size);
		};
		(callit $name: literal, $fun: ident, $type: ident, $size: literal, $its: literal) => {
			bench_it($name, |size| assert!($fun::<$type>(size).iter().all(|&x| x == 0 as $type)), $size, $its);
		};
		($type: ident, $size: literal) => {
			println!("\n{:?} x {:?}", $size, stringify!($type).replace("\"", ""));
			_bench_for_type!(call "Vec init", zero_initialized_vec_init, $type, $size);
			_bench_for_type!(call "Memset init", zero_initialized_memset, $type, $size);
			_bench_for_type!(call "Memset chunked init", zero_initialized_memset_chunked, $type, $size);
			_bench_for_type!(call "AVX init", zero_initialized_avx, $type, $size);
			_bench_for_type!(call "Fill init", zero_initialized_fill, $type, $size);
		};
		($type: ident, $size: literal, $its: literal) => {
			println!("\n{:?} x {:?}", $size, stringify!($type).replace("\"", ""));
			_bench_for_type!(callit "Vec init", zero_initialized_vec_init, $type, $size, $its);
			_bench_for_type!(callit "Memset init", zero_initialized_memset, $type, $size, $its);
			_bench_for_type!(callit "Memset chunked init", zero_initialized_memset_chunked, $type, $size, $its);
			_bench_for_type!(callit "AVX init", zero_initialized_avx, $type, $size, $its);
			_bench_for_type!(callit "Fill init", zero_initialized_fill, $type, $size, $its);
		};
	}
	pub fn _bench_zero_init() {
		_bench_for_type!(u8, 100_000, 100);
		_bench_for_type!(u8, 10_000_000, 100);
		_bench_for_type!(u16, 100_000, 100);
		_bench_for_type!(u16, 10_000_000, 100);
		_bench_for_type!(f32, 100_000, 100);
		_bench_for_type!(f32, 10_000_000, 100);
	}

	#[test]
	pub fn bench_zero_init() {
		_bench_zero_init();
	}
}


