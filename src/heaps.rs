use std::{
	cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd}, collections::{BinaryHeap, binary_heap::Iter}, iter::{IntoIterator, Iterator}
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


pub struct DualHeap<T: HeapKey, V: HeapValue> {
	min_heap: Vec<MinHeapPair<T,V>>,
	max_heap: Vec<MaxHeapPair<T,V>>,
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
	fn _cmp_idx<const MIN: bool>(&self, i: usize, j: usize) -> bool {
		if MIN { self.min_heap[i] > self.min_heap[j] } else { self.max_heap[i] > self.max_heap[j] }
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
	fn _pop_ends(&mut self) {
		self.size -= 1;
		self.min_heap.pop();
		self.max_heap.pop();
		self.min_idx.pop();
		self.max_idx.pop();
		debug_assert!(self.size == self.min_heap.len());
		debug_assert!(self.size == self.max_heap.len());
		debug_assert!(self.size == self.min_idx.len());
		debug_assert!(self.size == self.max_idx.len());
	}
	/// Pop either the maximum or the minimum value from the heap
	/// dependent on the flag MIN (true=min, false=max).
	/// Returns None if the heap is empty.
	#[inline(always)]
	pub fn pop<const MIN: bool>(&mut self) -> Option<(T,V)> {
		if self.size == 0 { return None; };
		Some(if self.size == 1 {
			let pair = &self.min_heap[0];
			let result = (pair.key, pair.value);
			self._pop_ends();
			result
		} else if MIN {
			let pair = &self.min_heap[0];
			let result = (pair.key, pair.value);
			let maxpos = self.min_idx[0];
			self._indexed_swap::<true>(0,self.size-1);
			self._indexed_swap::<false>(maxpos, self.size-1);
			self._pop_ends();
			self._sift_down::<true>(0);
			if maxpos < self.size {
				self._sift_up::<false>(maxpos);
				self._sift_down::<false>(maxpos);
			}
			result
		} else {
			let pair = &self.max_heap[0];
			let result = (pair.key, pair.value);
			let minpos = self.max_idx[0];
			self._indexed_swap::<false>(0,self.size-1);
			self._indexed_swap::<true>(minpos, self.size-1);
			self._pop_ends();
			self._sift_down::<false>(0);
			if minpos < self.size {
				self._sift_up::<true>(minpos);
				self._sift_down::<true>(minpos);
			}
			result
		})
	}
	#[inline(always)]
	fn _push_ends(&mut self, key: T, value: V) {
		self.min_heap.push(MinHeapPair::new(key, value));
		self.max_heap.push(MaxHeapPair::new(key, value));
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
		if self.size == 0 { return (key, value) };
		if MIN {
			let new_pair = MinHeapPair::new(key, value);
			let minpos = 0;
			if new_pair > self.min_heap[minpos] {
				(key, value)
			} else {
				let ret = (self.min_heap[minpos].key, self.min_heap[minpos].value);
				let maxpos = self.min_idx[minpos];
				self.min_heap[minpos] = new_pair;
				self.max_heap[maxpos] = MaxHeapPair::new(key, value);
				/* Move replaced element to the correct position in min heap */
				self._sift_down::<true>(minpos);
				/* Move replaced element to the correct position in max heap */
				self._sift_up::<false>(maxpos);
				self._sift_down::<false>(maxpos);
				ret
			}
		} else {
			let new_pair = MaxHeapPair::new(key, value);
			let maxpos = 0;
			if new_pair > self.max_heap[maxpos] {
				(key, value)
			} else {
				let ret = (self.max_heap[maxpos].key, self.max_heap[maxpos].value);
				let minpos = self.max_idx[maxpos];
				self.max_heap[maxpos] = new_pair;
				self.min_heap[minpos] = MinHeapPair::new(key, value);
				/* Move replaced element to the correct position in max heap */
				self._sift_down::<false>(maxpos);
				/* Move replaced element to the correct position in min heap */
				self._sift_up::<true>(minpos);
				self._sift_down::<true>(minpos);
				ret
			}
		}
	}
	/// Peek at the minimum or maximum value in the heap.
	/// Returns None if the heap is empty.
	#[inline(always)]
	pub fn peek<const MIN: bool>(&self) -> Option<(T,V)> {
		if self.size == 0 { return None; };
		Some(if MIN {
			let pair = &self.min_heap[0];
			(pair.key, pair.value)
		} else {
			let pair = &self.max_heap[0];
			(pair.key, pair.value)
		})
	}
	/// Create an iterator over the heap that pops the minimum or maximum value
	/// dependent on the flag MIN (true=min, false=max).
	#[inline(always)]
	pub fn into_iter<const MIN: bool>(self) -> DualHeapIter<MIN, T, V> {
		DualHeapIter::new(self)
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



pub trait GenericHeapPair: Ord {
	type Key: HeapKey;
	type Value: HeapValue;
	fn new(key: Self::Key, value: Self::Value) -> Self;
	fn key_ref<'a>(&'a self) -> &'a Self::Key;
	fn value_ref<'a>(&'a self) -> &'a Self::Value;
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
fn test_dual_heap() {
	let n = 10_000;
	type T = f64;
	let (data,sorted_data) = _make_data::<T>(n);
	let mut heap = DualHeap::<T,usize>::with_capacity(n);
	/* Test ascending order with pop min */
	data.iter().enumerate().for_each(|(i,&x)| heap.push(x,i));
	let result = heap.into_iter::<true>().map(|(k,_)|k).collect::<Vec<_>>();
	assert_eq!(result, sorted_data);
	/* Test descending order with pop max */
	let mut heap = DualHeap::<T,usize>::with_capacity(n);
	data.iter().enumerate().for_each(|(i,&x)| heap.push(x,i));
	let result = heap.into_iter::<false>().map(|(k,_)|k).collect::<Vec<_>>();
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
	assert_eq!(heap.peek::<true>().unwrap().0, _slice_min(&data[0..k]), "\nSlice: {:?}\nMin heap: {:?}\n", &data[0..k], heap.min_heap.iter().map(|x| x.key).collect::<Vec<_>>());
	/* Push the rest of the elements and pop min */
	(k..n).for_each(|i| {
		let (key,_) = heap.push_pop::<true>(data[i],i);
		assert!(key <= heap.min_heap.iter().map(|x| x.key).reduce(|a,b| if a<b {a} else {b}).unwrap());
	});
	/* Test push-pop max */
	let mut heap = DualHeap::<T,usize>::with_capacity(n);
	/* Push the first k elements */
	(0..k).for_each(|i| heap.push(data[i],i));
	assert_eq!(heap.peek::<false>().unwrap().0, _slice_max(&data[0..k]), "\nSlice: {:?}\nMin heap: {:?}\n", &data[0..k], heap.max_heap.iter().map(|x| x.key).collect::<Vec<_>>());
	/* Push the rest of the elements and pop min */
	(k..n).for_each(|i| {
		let (key,_) = heap.push_pop::<false>(data[i],i);
		assert!(key >= heap.max_heap.iter().map(|x| x.key).reduce(|a,b| if a<b {a} else {b}).unwrap());
	});
}


