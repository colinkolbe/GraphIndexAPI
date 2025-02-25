use std::{
	cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd}, collections::{binary_heap::Iter, BinaryHeap}, iter::{IntoIterator, Iterator}, mem::swap
};

pub trait HeapKey: PartialOrd+Copy {}
impl<T: PartialOrd+Copy> HeapKey for T {}
pub trait HeapValue: Copy {}
impl<T: Copy> HeapValue for T {}

pub trait GenericHeap: IntoIterator {
	type Key: HeapKey;
	type Value: HeapValue;
	type Pair: GenericHeapPair<Key=Self::Key, Value=Self::Value>;
	fn wrapped_heap(&self) -> &BinaryHeap<Self::Pair>;
	fn wrapped_heap_mut(&mut self) -> &mut BinaryHeap<Self::Pair>;
	#[inline(always)]
	fn push(&mut self, key: Self::Key, value: Self::Value) {
		self.wrapped_heap_mut().push(Self::Pair::new(key, value))
	}
	#[inline(always)]
	fn pop(&mut self) -> Option<(Self::Key, Self::Value)> {
		let element = self.wrapped_heap_mut().pop();
		if element.is_some() {
			unsafe {
				let element = element.unwrap_unchecked();
				Some((*element.key_ref(), *element.value_ref()))
			}
		} else { None }
	}
	#[inline(always)]
	fn peek(&self) -> Option<(Self::Key, Self::Value)> {
		let heap = self.wrapped_heap();
		let element = heap.peek();
		if element.is_some() {
			unsafe {
				let element = element.unwrap_unchecked();
				Some((*element.key_ref(), *element.value_ref()))
			}
		} else { None }
	}
	#[inline(always)]
	fn size(&self) -> usize {
		self.wrapped_heap().len()
	}
	#[inline(always)]
	fn reserve(&mut self, size: usize) {
		self.wrapped_heap_mut().reserve(size);
	}
	#[inline(always)]
	fn clear(&mut self) {
		self.wrapped_heap_mut().clear();
	}
	#[inline(always)]
	fn iter(&self) -> Iter<'_, Self::Pair> {
		self.wrapped_heap().iter()
	}
}
pub trait GenericHeapPair: Ord {
	type Key: HeapKey;
	type Value: HeapValue;
	fn new(key: Self::Key, value: Self::Value) -> Self;
	fn key_ref<'a>(&'a self) -> &'a Self::Key;
	fn value_ref<'a>(&'a self) -> &'a Self::Value;
}


pub mod old_impl {
	use crate::heaps::*;

	pub struct HeapIter<H: GenericHeap> {
		heap: H
	}
	impl<H: GenericHeap> HeapIter<H> {
		#[inline(always)]
		fn new(heap: H) -> Self { Self {heap: heap} }
	}
	impl<H: GenericHeap> Iterator for HeapIter<H> {
		type Item = (H::Key,H::Value);
		#[inline(always)]
		fn next(&mut self) -> Option<Self::Item> {
			self.heap.pop()
		}
	}

	pub struct MaxHeap<T: HeapKey, V: HeapValue> {
		heap: BinaryHeap<MaxHeapPair<T, V>>
	}
	impl<T: HeapKey, V: HeapValue> MaxHeap<T,V> {
		#[inline(always)]
		pub fn new() -> Self {
			MaxHeap{heap: BinaryHeap::new()}
		}
	}
	impl<T: HeapKey, V: HeapValue> GenericHeap for MaxHeap<T,V> {
		type Key = T;
		type Value = V;
		type Pair = MaxHeapPair<T,V>;
		#[inline(always)]
		fn wrapped_heap(&self) -> &BinaryHeap<MaxHeapPair<T,V>> { &self.heap }
		#[inline(always)]
		fn wrapped_heap_mut(&mut self) -> &mut BinaryHeap<MaxHeapPair<T,V>> { &mut self.heap }
	}
	impl<T: HeapKey, V: HeapValue> IntoIterator for MaxHeap<T,V> {
		type Item = (T,V);
		type IntoIter = HeapIter<Self>;
		#[inline(always)]
		fn into_iter(self) -> Self::IntoIter {
			HeapIter::new(self)
		}
	}


	pub struct MinHeap<T: HeapKey, V: HeapValue> {
		heap: BinaryHeap<MinHeapPair<T, V>>
	}
	impl<T: HeapKey, V: HeapValue> MinHeap<T,V> {
		#[inline(always)]
		pub fn new() -> Self {
			MinHeap{heap: BinaryHeap::new()}
		}
	}
	impl<T: HeapKey, V: HeapValue> GenericHeap for MinHeap<T,V> {
		type Key = T;
		type Value = V;
		type Pair = MinHeapPair<T,V>;
		#[inline(always)]
		fn wrapped_heap(&self) -> &BinaryHeap<MinHeapPair<T,V>> { &self.heap }
		#[inline(always)]
		fn wrapped_heap_mut(&mut self) -> &mut BinaryHeap<MinHeapPair<T,V>> { &mut self.heap }
	}
	impl<T: HeapKey, V: HeapValue> IntoIterator for MinHeap<T,V> {
		type Item = (T,V);
		type IntoIter = HeapIter<Self>;
		#[inline(always)]
		fn into_iter(self) -> Self::IntoIter {
			HeapIter::new(self)
		}
	}


	#[derive(Debug)]
	pub struct MinHeapPair<T: HeapKey, V: HeapValue> {
		key: T,
		value: V
	}
	impl<T: HeapKey, V: HeapValue> GenericHeapPair for MinHeapPair<T, V> {
		type Key = T;
		type Value = V;
		#[inline(always)]
		fn new(key: T, value: V) -> Self {
			MinHeapPair{key:key, value:value}
		}
		#[inline(always)]
		fn key_ref<'a>(&'a self) -> &'a T {&self.key}
		#[inline(always)]
		fn value_ref<'a>(&'a self) -> &'a V {&self.value}
	}
	impl<T: HeapKey, V: HeapValue> PartialEq for MinHeapPair<T, V> {
		#[inline(always)]
		fn eq(&self, other: &Self) -> bool {
			self.key == other.key
		}
	}
	impl<T: HeapKey, V: HeapValue> PartialOrd for MinHeapPair<T, V> {
		#[inline(always)]
		fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
			let pcmp = self.key.partial_cmp(&other.key);
			if pcmp.is_none() { None } else { unsafe { Some(pcmp.unwrap_unchecked().reverse()) } }
		}
	}
	impl<T: HeapKey, V: HeapValue> Eq for MinHeapPair<T, V> {}
	impl<T: HeapKey, V: HeapValue> Ord for MinHeapPair<T, V> {
		#[inline(always)]
		fn cmp(&self, other: &Self) -> Ordering {
			unsafe { self.key.partial_cmp(&other.key).unwrap_unchecked() }
		}
	}
	
	#[derive(Debug)]
	pub struct MaxHeapPair<T: HeapKey, V: HeapValue> {
		key: T,
		value: V
	}
	impl<T: HeapKey, V: HeapValue> GenericHeapPair for MaxHeapPair<T, V> {
		type Key = T;
		type Value = V;
		#[inline(always)]
		fn new(key: T, value: V) -> Self {
			MaxHeapPair{key:key, value:value}
		}
		#[inline(always)]
		fn key_ref<'a>(&'a self) -> &'a T {&self.key}
		#[inline(always)]
		fn value_ref<'a>(&'a self) -> &'a V {&self.value}
	}
	impl<T: HeapKey, V: HeapValue> PartialEq for MaxHeapPair<T, V> {
		#[inline(always)]
		fn eq(&self, other: &Self) -> bool {
			self.key == other.key
		}
	}
	impl<T: HeapKey, V: HeapValue> PartialOrd for MaxHeapPair<T, V> {
		#[inline(always)]
		fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
			self.key.partial_cmp(&other.key)
		}
	}
	impl<T: HeapKey, V: HeapValue> Eq for MaxHeapPair<T, V> {}
	impl<T: HeapKey, V: HeapValue> Ord for MaxHeapPair<T, V> {
		#[inline(always)]
		fn cmp(&self, other: &Self) -> Ordering {
			unsafe { self.key.partial_cmp(&other.key).unwrap_unchecked() }
		}
	}
}

pub mod new_impl {
	use std::mem::swap;
	use crate::heaps::*;

	pub struct GenericHeap<T: HeapKey, V: HeapValue, const MIN: bool> {
		heap: Vec<(T,V)>,
	}
	impl<T: HeapKey, V: HeapValue, const MIN: bool> GenericHeap<T,V,MIN> {
		/// Creates a new empty heap
		#[inline(always)]
		pub fn new() -> Self {
			GenericHeap{heap: Vec::new()}
		}
		/// Creates a new empty heap with a given preallocated capacity
		#[inline(always)]
		pub fn with_capacity(capacity: usize) -> Self {
			GenericHeap{heap: Vec::with_capacity(capacity)}
		}
		/// Internal helper function to compare two keys
		#[inline(always)]
		fn _cmp(&self, a: T, b: T) -> bool {
			if MIN { a < b } else { a > b }
		}
		/// Internal helper function to compare two elements by index using their key
		#[inline(always)]
		fn _cmp_idx(&self, i: usize, j: usize) -> bool {
			self._cmp(self.heap[i].0, self.heap[j].0)
		}
		/// Internal helper function to swap two elements by their index
		#[inline(always)]
		fn _indexed_swap(&mut self, i: usize, j: usize) {
			debug_assert!(i < self.heap.len());
			self.heap.swap(i,j);
		}
		/// Internal helper function to sift down an element at a given position
		#[inline(always)]
		fn _sift_down(&mut self, mut pos: usize) {
			let n = self.heap.len();
			let mut lc = ((pos+1)<<1)-1;
			let mut rc = lc+1;
			loop {
				/* No children */
				if lc >= n { return; }
				/* Select child to use for sifting */
				let prio_child = if rc >= n || self._cmp_idx(lc,rc) { lc } else { rc };
				/* Child is "larger" than parent, swap and keep sifting */
				if self._cmp_idx(prio_child,pos) {
					self._indexed_swap(pos,prio_child);
					pos = prio_child;
				} else {
					/* Child is "smaller" than parent, we are done */
					break;
				}
				/* Update child pointer */
				lc = ((pos+1)<<1)-1;
				rc = lc+1;
			}
		}
		/// Internal helper function to sift up an element at a given position
		#[inline(always)]
		fn _sift_up(&mut self, mut pos: usize) {
			while pos > 0 {
				let parent = (pos-1)>>1;
				if self._cmp_idx(pos,parent) {
					self._indexed_swap(pos,parent);
					pos = parent;
				} else {
					break;
				}
			}
		}
		/// Internal helper function to heapify the heap
		#[inline(always)]
		fn _heapify(&mut self) {
			let n = self.heap.len();
			(1..n).for_each(|i| self._sift_up(i));
		}
		/// Internal helper function to pop the last element from the heap
		#[inline(always)]
		fn _pop_end(&mut self) -> Option<(T,V)> {
			self.heap.pop()
		}
		/// Pop the minimum or maximum value from the heap, returns None if empty
		#[inline(always)]
		pub fn pop(&mut self) -> Option<(T,V)> {
			let n = self.heap.len();
			if n == 0 {
				None
			} else if n == 1 {
				self._pop_end()
			} else {
				self._indexed_swap(0,n-1);
				let pair = self._pop_end();
				self._sift_down(0);
				pair
			}
		}
		/// Internal helper function to push a new key-value pair to the end of the heap
		#[inline(always)]
		fn _push_end(&mut self, key: T, value: V) {
			self.heap.push((key, value));
		}
		/// Push a new key-value pair into the heap
		#[inline(always)]
		pub fn push(&mut self, key: T, value: V) {
			let n = self.heap.len();
			self._push_end(key, value);
			self._sift_up(n);
		}
		/// Simulate a push followed by a pop of the minimum or maximum value.
		#[inline(always)]
		pub fn push_pop(&mut self, key: T, value: V) -> (T,V) {
			let n = self.heap.len();
			if n == 0 || self._cmp(key, self.heap[0].0) {
				(key, value)
			} else {
				let mut new_pair = (key, value);
				swap(&mut new_pair, unsafe{self.heap.get_unchecked_mut(0)});
				self._sift_down(0);
				new_pair
			}
		}
		/// Peek at the minimum or maximum value in the heap.
		#[inline(always)]
		pub fn peek(&self) -> Option<(T,V)> {
			if self.heap.len() == 0 {
				None
			} else {
				Some(self.heap[0])
			}
		}
		/// Create an iterator over the heap that pops the minimum or maximum value
		#[inline(always)]
		pub fn into_sorted_iter(self) -> GenericHeapIter<T,V,MIN> {
			GenericHeapIter::new(self)
		}
		/// Create an iterator over the heap in memory-aligned order (not sorted)
		#[inline(always)]
		pub fn into_iter(self) -> std::vec::IntoIter<(T,V)> {
			self.heap.into_iter()
		}
		/// Create an iterator over the heap in sorted order (empties the heap)
		#[inline(always)]
		pub fn sorted_iter(&mut self) -> GenericHeapRefIter<T,V,MIN> {
			GenericHeapRefIter::new(self)
		}
		/// Create an iterator over the heap in memory-aligned order (not sorted but does not change the heap)
		#[inline(always)]
		pub fn iter(&self) -> std::slice::Iter<(T,V)> {
			self.heap.iter()
		}
		/// Create a mutable iterator over the heap in memory-aligned order (not sorted but does not change the heap)
		/// If any keys are changed, the heap becomes invalid!
		#[inline(always)]
		pub fn iter_mut(&mut self) -> core::slice::IterMut<(T,V)> {
			self.heap.iter_mut()
		}
		/// Get the number of elements in the heap
		#[inline(always)]
		pub fn size(&self) -> usize {
			self.heap.len()
		}
		/// Reserved additional capacity for the heap akin to std::vec::Vec::reserve
		#[inline(always)]
		pub fn reserve(&mut self, capacity: usize) {
			self.heap.reserve(capacity);
		}
		/// Clear the heap
		#[inline(always)]
		pub fn clear(&mut self) {
			self.heap.clear();
		}
		#[inline(always)]
		pub fn as_slice(&self) -> &[(T,V)] {
			self.heap.as_slice()
		}
		#[inline(always)]
		pub fn as_mut_slice(&mut self) -> &mut [(T,V)] {
			self.heap.as_mut_slice()
		}
		/// Removes and returns the element at index `idx` or None if the index is out of bounds.
		/// The index describes the position in the underlying vector, not the heap order.
		/// The index should thus be fetched by using the `iter` function.
		/// The heap is restructured after the removal.
		#[inline(always)]
		pub fn remove_by_idx(&mut self, idx: usize) -> Option<(T,V)> {
			let n = self.size();
			if n == 0 || idx >= n { None } else {
				self._indexed_swap(idx, self.size()-1);
				let result = self._pop_end();
				self._sift_down(idx);
				result
			}
		}
		/// Unsafe version of `remove_by_idx` that does not check the index bounds.
		#[inline(always)]
		pub fn remove_by_idx_unchecked(&mut self, idx: usize) -> (T,V) {
			self._indexed_swap(idx, self.size()-1);
			let result = self._pop_end();
			self._sift_down(idx);
			unsafe{result.unwrap_unchecked()}
		}
		/// Updates the key of an element at index `idx` or returns false if the index is out of bounds.
		/// The index describes the position in the underlying vector, not the heap order.
		/// The index should thus be fetched by using the `iter` function.
		/// The heap is restructured after the update.
		#[inline(always)]
		pub fn update_key(&mut self, idx: usize, key: T) -> bool {
			let n = self.size();
			if n == 0 || idx >= n { false } else {
				unsafe {
					let sift_up = if MIN { self.heap.get_unchecked(idx).0 > key } else { self.heap.get_unchecked(idx).0 < key };
					self.heap.get_unchecked_mut(idx).0 = key;
					if sift_up { self._sift_up(idx); } else { self._sift_down(idx); }
				}
				true
			}
		}
		/// Unsafe version of `update_key` that does not check the index bounds.
		#[inline(always)]
		pub fn update_key_unchecked(&mut self, idx: usize, key: T) {
			unsafe {
				let sift_up = if MIN { self.heap.get_unchecked(idx).0 > key } else { self.heap.get_unchecked(idx).0 < key };
				self.heap.get_unchecked_mut(idx).0 = key;
				if sift_up { self._sift_up(idx); } else { self._sift_down(idx); }
			}
		}
	}
	
	pub struct GenericHeapIter<T: HeapKey, V: HeapValue, const MIN: bool> {
		heap: GenericHeap<T,V,MIN>
	}
	impl<T: HeapKey, V: HeapValue, const MIN: bool> GenericHeapIter<T, V, MIN> {
		#[inline(always)]
		fn new(heap: GenericHeap<T,V,MIN>) -> Self { Self {heap: heap} }
	}
	impl<T: HeapKey, V: HeapValue, const MIN: bool> Iterator for GenericHeapIter<T, V, MIN> {
		type Item = (T,V);
		#[inline(always)]
		fn next(&mut self) -> Option<Self::Item> {
			self.heap.pop()
		}
	}
	pub struct GenericHeapRefIter<'a, T: HeapKey, V: HeapValue, const MIN: bool> {
		heap: &'a mut GenericHeap<T,V,MIN>
	}
	impl<'a, T: HeapKey, V: HeapValue, const MIN: bool> GenericHeapRefIter<'a, T, V, MIN> {
		#[inline(always)]
		fn new(heap: &'a mut GenericHeap<T,V,MIN>) -> Self { Self {heap: heap} }
	}
	impl<'a, T: HeapKey, V: HeapValue, const MIN: bool> Iterator for GenericHeapRefIter<'a, T, V, MIN> {
		type Item = (T,V);
		#[inline(always)]
		fn next(&mut self) -> Option<Self::Item> {
			self.heap.pop()
		}
	}

	pub type MinHeap<T,V> = GenericHeap<T,V,true>;
	pub type MaxHeap<T,V> = GenericHeap<T,V,false>;
}

pub type MinHeap<T,V> = new_impl::MinHeap<T,V>;
pub type MaxHeap<T,V> = new_impl::MaxHeap<T,V>;

pub struct DualHeap<T: HeapKey, V: HeapValue> {
	min_heap: Vec<(T,V)>,
	max_heap: Vec<(T,V)>,
	min_idx: Vec<usize>,
	max_idx: Vec<usize>,
	size: usize,
}
impl<T: HeapKey, V: HeapValue> DualHeap<T,V> {
	/// Create a new empty dual heap
	#[inline(always)]
	pub fn new() -> Self {
		/* Using boxed slices instead of vec would be smaller but its not really worth the headache */
		DualHeap{
			min_heap: Vec::new(),
			max_heap: Vec::new(),
			min_idx: Vec::new(),
			max_idx: Vec::new(),
			size: 0,
		}
	}
	/// Create a new empty dual heap with a given capacity
	#[inline(always)]
	pub fn with_capacity(capacity: usize) -> Self {
		/* Using boxed slices instead of vec would be smaller but its not really worth the headache */
		DualHeap{
			min_heap: Vec::with_capacity(capacity),
			max_heap: Vec::with_capacity(capacity),
			min_idx: Vec::with_capacity(capacity),
			max_idx: Vec::with_capacity(capacity),
			size: 0,
		}
	}
	#[inline(always)]
	fn _cmp<const MIN: bool>(&self, a: T, b: T) -> bool {
		if MIN { a < b } else { a > b }
	}
	#[inline(always)]
	fn _cmp_idx<const MIN: bool>(&self, i: usize, j: usize) -> bool {
		if MIN { self.min_heap[i].0 < self.min_heap[j].0 } else { self.max_heap[i].0 > self.max_heap[j].0 }
	}
	#[inline(always)]
	fn _indexed_swap<const MIN: bool>(&mut self, i: usize, j: usize) {
		debug_assert!(i < self.size);
		debug_assert!(j < self.size);
		if MIN {
			self.min_heap.swap(i,j);
			self.min_idx.swap(i,j);
			self.max_idx[self.min_idx[i]] = i;
			self.max_idx[self.min_idx[j]] = j;
		} else {
			self.max_heap.swap(i,j);
			self.max_idx.swap(i,j);
			self.min_idx[self.max_idx[i]] = i;
			self.min_idx[self.max_idx[j]] = j;
		};
	}
	#[inline(always)]
	fn _sift_down<const MIN: bool>(&mut self, mut pos: usize) {
		let n = self.size;
		let mut lc = ((pos+1)<<1)-1;
		let mut rc = lc+1;
		loop {
			/* No children */
			if lc >= n { return; }
			/* Select child to use for sifting */
			let prio_child = if rc >= n || self._cmp_idx::<MIN>(lc,rc) { lc } else { rc };
			/* Child is "larger" than parent, swap and keep sifting */
			if self._cmp_idx::<MIN>(prio_child,pos) {
				self._indexed_swap::<MIN>(pos,prio_child);
				pos = prio_child;
			} else {
				/* Child is "smaller" than parent, we are done */
				break;
			}
			/* Update child pointer */
			lc = ((pos+1)<<1)-1;
			rc = lc+1;
		}
	}
	#[inline(always)]
	fn _sift_up<const MIN: bool>(&mut self, mut pos: usize) {
		while pos > 0 {
			let parent = (pos-1)>>1;
			if self._cmp_idx::<MIN>(pos,parent) {
				self._indexed_swap::<MIN>(pos,parent);
				pos = parent;
			} else {
				break;
			}
		}
	}
	#[inline(always)]
	fn _heapify<const MIN: bool>(&mut self) {
		let n = self.size;
		(1..n).for_each(|i| self._sift_up::<MIN>(i));
	}
	#[inline(always)]
	fn _pop_ends(&mut self) -> (Option<(T,V)>,Option<(T,V)>) {
		self.size -= 1;
		let result = (self.min_heap.pop(),self.max_heap.pop());
		self.min_idx.pop();
		self.max_idx.pop();
		debug_assert!(self.size == self.min_heap.len());
		debug_assert!(self.size == self.max_heap.len());
		debug_assert!(self.size == self.min_idx.len());
		debug_assert!(self.size == self.max_idx.len());
		result
	}
	/// Pop either the maximum or the minimum value from the heap
	/// dependent on the flag MIN (true=min, false=max).
	/// Returns None if the heap is empty.
	#[inline(always)]
	pub fn pop<const MIN: bool>(&mut self) -> Option<(T,V)> {
		if self.size == 0 { return None; };
		if self.size == 1 {
			self._pop_ends().0
		} else if MIN {
			let maxpos = self.min_idx[0];
			self._indexed_swap::<true>(0,self.size-1);
			self._indexed_swap::<false>(maxpos, self.size-1);
			let result = self._pop_ends().0;
			self._sift_down::<true>(0);
			if maxpos < self.size {
				self._sift_up::<false>(maxpos);
				self._sift_down::<false>(maxpos);
			}
			result
		} else {
			let minpos = self.max_idx[0];
			self._indexed_swap::<false>(0,self.size-1);
			self._indexed_swap::<true>(minpos, self.size-1);
			let result = self._pop_ends().1;
			self._sift_down::<false>(0);
			if minpos < self.size {
				self._sift_up::<true>(minpos);
				self._sift_down::<true>(minpos);
			}
			result
		}
	}
	#[inline(always)]
	fn _push_ends(&mut self, key: T, value: V) {
		self.min_heap.push((key, value));
		self.max_heap.push((key, value));
		self.min_idx.push(self.size);
		self.max_idx.push(self.size);
		self.size += 1;
		debug_assert!(self.size == self.min_heap.len());
		debug_assert!(self.size == self.max_heap.len());
		debug_assert!(self.size == self.min_idx.len());
		debug_assert!(self.size == self.max_idx.len());
	}
	/// Push a new key-value pair into the heap
	#[inline(always)]
	pub fn push(&mut self, key: T, value: V) {
		let n = self.size;
		self._push_ends(key, value);
		debug_assert!(self.min_idx[n] == n);
		debug_assert!(self.max_idx[n] == n);
		self._sift_up::<true>(n);
		self._sift_up::<false>(n);
	}
	/// Simulate a push followed by a pop of the minimum or maximum value.
	/// This is more efficient than calling both functions separately.
	#[inline(always)]
	pub fn push_pop<const MIN: bool>(&mut self, key: T, value: V) -> (T,V) {
		if self.size == 0 {
			(key, value)
		} else if MIN {
			let minpos = 0;
			if self._cmp::<true>(key, self.min_heap[minpos].0) {
				(key, value)
			} else {
				let mut new_pair = (key, value);
				swap(&mut new_pair, unsafe{self.min_heap.get_unchecked_mut(minpos)});
				let maxpos = self.min_idx[minpos];
				unsafe{*self.max_heap.get_unchecked_mut(maxpos) = (key, value);}
				/* Move replaced element to the correct position in min heap */
				self._sift_down::<true>(minpos);
				/* Move replaced element to the correct position in max heap */
				self._sift_up::<false>(maxpos);
				self._sift_down::<false>(maxpos);
				new_pair
			}
		} else {
			let maxpos = 0;
			if self._cmp::<false>(key, self.max_heap[maxpos].0) {
				(key, value)
			} else {
				let mut new_pair = (key, value);
				swap(&mut new_pair, unsafe{self.max_heap.get_unchecked_mut(maxpos)});
				let minpos = self.max_idx[maxpos];
				unsafe{*self.min_heap.get_unchecked_mut(minpos) = (key, value);}
				/* Move replaced element to the correct position in max heap */
				self._sift_down::<false>(maxpos);
				/* Move replaced element to the correct position in min heap */
				self._sift_up::<true>(minpos);
				self._sift_down::<true>(minpos);
				new_pair
			}
		}
	}
	/// Peek at the minimum or maximum value in the heap.
	/// Returns None if the heap is empty.
	#[inline(always)]
	pub fn peek<const MIN: bool>(&self) -> Option<(T,V)> {
		if self.size == 0 {
			None
		} else if MIN {
			Some(self.min_heap[0])
		} else {
			Some(self.max_heap[0])
		}
	}
	/// Create an iterator over the heap that pops the minimum or maximum value
	/// dependent on the flag MIN (true=min, false=max).
	#[inline(always)]
	pub fn into_sorted_iter<const MIN: bool>(self) -> DualHeapIter<MIN, T, V> {
		DualHeapIter::new(self)
	}
	#[inline(always)]
	pub fn into_iter<const MIN: bool>(self) -> std::vec::IntoIter<(T,V)> {
		if MIN {
			self.min_heap.into_iter()
		} else {
			self.max_heap.into_iter()
		}
	}
	#[inline(always)]
	pub fn iter<const MIN: bool>(&self) -> std::slice::Iter<(T,V)> {
		if MIN {
			self.min_heap.iter()
		} else {
			self.max_heap.iter()
		}
	}
	#[inline(always)]
	pub fn iter_mut<const MIN: bool>(&mut self) -> std::slice::IterMut<(T,V)> {
		if MIN {
			self.min_heap.iter_mut()
		} else {
			self.max_heap.iter_mut()
		}
	}
	/// Get the number of elements in the heap
	#[inline(always)]
	pub fn size(&self) -> usize {
		self.size
	}
	/// Reserved additional capacity for the heap akin to std::vec::Vec::reserve
	#[inline(always)]
	pub fn reserve(&mut self, capacity: usize) {
		self.min_heap.reserve(capacity);
		self.max_heap.reserve(capacity);
		self.min_idx.reserve(capacity);
		self.max_idx.reserve(capacity);
	}
	#[inline(always)]
	pub fn clear(&mut self) {
		self.min_heap.clear();
		self.max_heap.clear();
		self.min_idx.clear();
		self.max_idx.clear();
		self.size = 0;
	}
	#[inline(always)]
	pub fn as_slice<const MIN: bool>(&self) -> &[(T,V)] {
		if MIN {
			self.min_heap.as_slice()
		} else {
			self.max_heap.as_slice()
		}
	}
	#[inline(always)]
	pub fn as_mut_slice<const MIN: bool>(&mut self) -> &mut [(T,V)] {
		if MIN {
			self.min_heap.as_mut_slice()
		} else {
			self.max_heap.as_mut_slice()
		}
	}
}
pub struct DualHeapIter<const MIN: bool, T: HeapKey, V: HeapValue> {
	heap: DualHeap<T,V>
}
impl<const MIN: bool, T: HeapKey, V: HeapValue> DualHeapIter<MIN, T, V> {
	#[inline(always)]
	fn new(heap: DualHeap<T,V>) -> Self { Self {heap: heap} }
}
impl<const MIN: bool, T: HeapKey, V: HeapValue> Iterator for DualHeapIter<MIN, T, V> {
	type Item = (T,V);
	#[inline(always)]
	fn next(&mut self) -> Option<Self::Item> {
		self.heap.pop::<MIN>()
	}
}





#[cfg(test)]
fn _make_data<T: Clone+PartialOrd>(n: usize) -> (Vec<T>,Vec<T>) where ndarray_rand::rand_distr::Standard: ndarray_rand::rand_distr::Distribution<T> {
	use ndarray_rand::rand;
	let data: Vec<T> = (0..n).map(|_| rand::random()).collect();
	let mut sorted_data = data.clone();
	sorted_data.sort_by(|a,b| a.partial_cmp(b).unwrap());
	(data,sorted_data)
}
#[cfg(test)]
fn _sort_slice<T: Clone+PartialOrd>(data: &[T]) -> Vec<T>{
	let mut data = data.to_vec();
	data.sort_by(|a,b| a.partial_cmp(b).unwrap());
	data
}
#[cfg(test)]
fn _slice_max<T: PartialOrd+Copy>(data: &[T]) -> T {
	data.iter().fold(data[0], |acc, &x| if x > acc { x } else { acc })
}
#[cfg(test)]
fn _slice_min<T: PartialOrd+Copy>(data: &[T]) -> T {
	data.iter().fold(data[0], |acc, &x| if x < acc { x } else { acc })
}
#[test]
fn test_min_heap() {
	let n = 10_000;
	type T = f64;
	let (data,sorted_data) = _make_data::<T>(n);
	let mut heap = MinHeap::<T,usize>::new();
	data.iter().enumerate().for_each(|(i,&x)| heap.push(x,i));
	let result = (0..n).map(|_| heap.pop().unwrap().0).collect::<Vec<_>>();
	assert_eq!(result, sorted_data);
}
#[test]
fn test_max_heap() {
	let n = 10_000;
	type T = f64;
	let (data,sorted_data) = _make_data::<T>(n);
	let sorted_data = sorted_data.into_iter().rev().collect::<Vec<_>>();
	let mut heap = MaxHeap::<T,usize>::new();
	data.iter().enumerate().for_each(|(i,&x)| heap.push(x,i));
	let result = (0..n).map(|_| heap.pop().unwrap().0).collect::<Vec<_>>();
	assert_eq!(result, sorted_data);
}
#[test]
fn benchmark_compare_min_heap() {
	type Old<T,V> = old_impl::MinHeap<T,V>;
	type New<T,V> = new_impl::MinHeap<T,V>;
	let n = 10_000_000;
	type T = f64;
	let (data,_) = _make_data::<T>(n);
	let mut old_heap = Old::<T,usize>::new();
	let mut new_heap = New::<T,usize>::new();
	let old_time = std::time::Instant::now();
	old_heap.reserve(n);
	data.iter().enumerate().for_each(|(i,v)| { old_heap.push(*v,i); });
	let old_time = old_time.elapsed();
	let new_time = std::time::Instant::now();
	new_heap.reserve(n);
	data.iter().enumerate().for_each(|(i,v)| { new_heap.push(*v,i); });
	let new_time = new_time.elapsed();
	println!("Pushing: Old: {:?}, New: {:?}", old_time, new_time);
	let old_time = std::time::Instant::now();
	let old_result = (0..n).map(|_| old_heap.pop().unwrap().0).collect::<Vec<_>>();
	let old_time = old_time.elapsed();
	let new_time = std::time::Instant::now();
	let new_result = (0..n).map(|_| new_heap.pop().unwrap().0).collect::<Vec<_>>();
	let new_time = new_time.elapsed();
	assert_eq!(old_result, new_result);
	println!("Popping: Old: {:?}, New: {:?}", old_time, new_time);
	let max_size = 100;
	let old_time = std::time::Instant::now();
	let mut old_result = Vec::new();
	data.iter().enumerate().for_each(|(i,&v)| {
		if old_heap.size() == max_size {
			if v < old_heap.peek().unwrap().0 {
				old_result.push(v);
			} else {
				old_result.push(old_heap.pop().unwrap().0);
				old_heap.push(v,i);
			}
		} else {
			old_heap.push(v,i);
		}
	});
	let old_time = old_time.elapsed();
	let new_time = std::time::Instant::now();
	let mut new_result = Vec::new();
	data.iter().enumerate().for_each(|(i,&v)| {
		if new_heap.size() == max_size {
			new_result.push(new_heap.push_pop(v, i).0);
		} else {
			new_heap.push(v,i);
		}
	});
	let new_time = new_time.elapsed();
	assert_eq!(old_result, new_result);
	println!("Push-pop: Old: {:?}, New: {:?}", old_time, new_time);
}
#[test]
fn benchmark_compare_max_heap() {
	type Old<T,V> = old_impl::MaxHeap<T,V>;
	type New<T,V> = new_impl::MaxHeap<T,V>;
	let n = 10_000_000;
	type T = f64;
	let (data,_) = _make_data::<T>(n);
	let mut old_heap = Old::<T,usize>::new();
	let mut new_heap = New::<T,usize>::new();
	let old_time = std::time::Instant::now();
	old_heap.reserve(n);
	data.iter().enumerate().for_each(|(i,v)| { old_heap.push(*v,i); });
	let old_time = old_time.elapsed();
	let new_time = std::time::Instant::now();
	new_heap.reserve(n);
	data.iter().enumerate().for_each(|(i,v)| { new_heap.push(*v,i); });
	let new_time = new_time.elapsed();
	println!("Pushing: Old: {:?}, New: {:?}", old_time, new_time);
	let old_time = std::time::Instant::now();
	let old_result = (0..n).map(|_| old_heap.pop().unwrap().0).collect::<Vec<_>>();
	let old_time = old_time.elapsed();
	let new_time = std::time::Instant::now();
	let new_result = (0..n).map(|_| new_heap.pop().unwrap().0).collect::<Vec<_>>();
	let new_time = new_time.elapsed();
	assert_eq!(old_result, new_result);
	println!("Popping: Old: {:?}, New: {:?}", old_time, new_time);
	let max_size = 100;
	let old_time = std::time::Instant::now();
	let mut old_result = Vec::new();
	data.iter().enumerate().for_each(|(i,&v)| {
		if old_heap.size() == max_size {
			if v > old_heap.peek().unwrap().0 {
				old_result.push(v);
			} else {
				old_result.push(old_heap.pop().unwrap().0);
				old_heap.push(v,i);
			}
		} else {
			old_heap.push(v,i);
		}
	});
	let old_time = old_time.elapsed();
	let new_time = std::time::Instant::now();
	let mut new_result = Vec::new();
	data.iter().enumerate().for_each(|(i,&v)| {
		if new_heap.size() == max_size {
			new_result.push(new_heap.push_pop(v, i).0);
		} else {
			new_heap.push(v,i);
		}
	});
	let new_time = new_time.elapsed();
	assert_eq!(old_result, new_result);
	println!("Push-pop: Old: {:?}, New: {:?}", old_time, new_time);
}
#[test]
fn test_dual_heap() {
	let n = 10_000;
	type T = f64;
	let (data,sorted_data) = _make_data::<T>(n);
	let mut heap = DualHeap::<T,usize>::with_capacity(n);
	/* Test comparing function */
	(0..data.len()-1).map(|i| (data[i],data[i+1])).for_each(|(a,b)| {
		assert!(heap._cmp::<true>(a,b) == (a < b), "a: {:?}, b: {:?}", a, b);
		assert!(heap._cmp::<false>(a,b) == (a > b), "a: {:?}, b: {:?}", a, b);
	});
	/* Test ascending order with pop min */
	data.iter().enumerate().for_each(|(i,&x)| heap.push(x,i));
	let result = heap.into_sorted_iter::<true>().map(|(k,_)|k).collect::<Vec<_>>();
	assert_eq!(result, sorted_data);
	/* Test descending order with pop max */
	let mut heap = DualHeap::<T,usize>::with_capacity(n);
	data.iter().enumerate().for_each(|(i,&x)| heap.push(x,i));
	let result = heap.into_sorted_iter::<false>().map(|(k,_)|k).collect::<Vec<_>>();
	assert_eq!(result, sorted_data.iter().map(|&v|v).rev().collect::<Vec<_>>());
	/* Test alternating order wirth pop min and pop max */
	let mut heap = DualHeap::<T,usize>::with_capacity(n);
	data.iter().enumerate().for_each(|(i,&x)| heap.push(x,i));
	let mut result = vec![0 as T; n];
	let mut i = 0;
	let mut j = n-1;
	while heap.size() > 0 {
		result[i] = heap.pop::<true>().unwrap().0;
		i += 1;
		if heap.size() > 0 {
			result[j] = heap.pop::<false>().unwrap().0;
			j -= 1;
		}
	}
	assert_eq!(result, sorted_data);
}
#[test]
fn test_dual_heap_push_pop() {
	let n = 10_000;
	let k = 15;
	type T = f64;
	let (data,_sorted_data) = _make_data::<T>(n);
	/* Test push-pop min */
	let mut heap = DualHeap::<T,usize>::with_capacity(n);
	/* Push the first k elements */
	(0..k).for_each(|i| heap.push(data[i],i));
	assert_eq!(heap.peek::<true>().unwrap().0, _slice_min(&data[0..k]), "\nSlice: {:?}\nMin heap: {:?}\n", &data[0..k], heap.min_heap.iter().map(|x| x.0).collect::<Vec<_>>());
	/* Push the rest of the elements and pop min */
	(k..n).for_each(|i| {
		let (key,_) = heap.push_pop::<true>(data[i],i);
		assert!(key <= heap.iter::<true>().map(|x| x.0).reduce(|a,b| if a<b {a} else {b}).unwrap());
	});
	/* Test push-pop max */
	let mut heap = DualHeap::<T,usize>::with_capacity(n);
	/* Push the first k elements */
	(0..k).for_each(|i| heap.push(data[i],i));
	assert_eq!(heap.peek::<false>().unwrap().0, _slice_max(&data[0..k]), "\nSlice: {:?}\nMin heap: {:?}\n", &data[0..k], heap.max_heap.iter().map(|x| x.0).collect::<Vec<_>>());
	/* Push the rest of the elements and pop min */
	(k..n).for_each(|i| {
		let (key,_) = heap.push_pop::<false>(data[i],i);
		assert!(key >= heap.iter::<false>().map(|x| x.0).reduce(|a,b| if a>b {a} else {b}).unwrap());
	});
}


