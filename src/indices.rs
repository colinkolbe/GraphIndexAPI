use core::panic;
use std::collections::HashSet;

use ndarray::{Data, Array1, Array2, ArrayBase, Ix1, Ix2};

use crate::data::MatrixDataSource;
use crate::graphs::Graph;
use crate::heaps::{DualHeap, GenericHeap, GenericHeapPair, MaxHeap, MaxHeapPair, MinHeap};
use crate::random::random_unique_uint;
use crate::types::{UnsignedInteger, Float, trait_combiner};
use crate::measures::Distance;

pub trait IndexedDistance<R: UnsignedInteger, F: Float, Dist: Distance<F>>: MatrixDataSource<F> {
	fn distance<D1: Data<Elem=F>, D2: Data<Elem=F>>(&self, i: &ArrayBase<D1, Ix1>, j: &ArrayBase<D2, Ix1>) -> F;
	fn half_indexed_distance<D: Data<Elem=F>>(&self, i: R, q: &ArrayBase<D, Ix1>) -> F {
		unsafe { self.distance(&self.get_row(i.to_usize().unwrap_unchecked()), q) }
	}
	fn indexed_distance(&self, i: R, j: R) -> F {
		unsafe { self.distance(&self.get_row(i.to_usize().unwrap_unchecked()), &self.get_row(j.to_usize().unwrap_unchecked())) }
	}
}
pub trait RangeIndex<R: UnsignedInteger, F: Float, Dist: Distance<F>> {
	fn range_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix1>, range: F) -> (Array1<R>, Array1<F>);
	fn range_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix2>, range: F) -> (Vec<Array1<R>>, Vec<Array1<F>>);
}
pub trait KnnIndex<R: UnsignedInteger, F: Float, Dist: Distance<F>> {
	fn knn_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix1>, k: usize) -> (Array1<R>, Array1<F>);
	fn knn_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D, Ix2>, k: usize) -> (Array2<R>, Array2<F>);
}
trait_combiner!(GeneralIndex[R: UnsignedInteger, F: Float, Dist: (Distance<F>)]: (RangeIndex<R, F, Dist>) + (KnnIndex<R, F, Dist>) + (IndexedDistance<R, F, Dist>) + (MatrixDataSource<F>));





#[derive(Debug, Clone)]
pub struct NoSuchLayerError;
pub trait GraphIndex<R: UnsignedInteger, F: Float, Dist: Distance<F>>: GeneralIndex<R, F, Dist> {
	/// Returns the number of layers in the graph index.
	fn layer_count(&self) -> usize;
	/// Returns the graph at the given layer if available, otherwise returns an error.
	/// 0 is the lowest level, layer_count()-1 is the highest level.
	fn get_layer(&self, layer: usize) -> Result<&dyn Graph<R>, NoSuchLayerError>;
	/// Returns a map of the local graph node IDs to the global data IDs.
	/// If the layer does not have a mapping (i.e. local IDs are global IDs), None is returned.
	fn get_global_layer_ids(&self, layer: usize) -> Option<&Vec<R>>;
	/// Returns a map of the local graph node IDs to the local graph node IDs of the next layer.
	/// If the layer does not have a mapping (i.e. local IDs are identical), None is returned.
	fn get_local_layer_ids(&self, layer: usize) -> Option<&Vec<R>>;
	/// Initializes a maximum heap with random vertices from the graph and distances to the query.
	/// `ids` is a vector of the global sample indices from the graph or None in which case
	/// indices are sampled from 0 to N (exclusive) where N is the size of the covered data.
	/// Distances are computed to the query vector `q`.
	fn _random_init_maxheap<D: Data<Elem=F>>(&self, ids: Option<&Vec<R>>, k: usize, q: &ArrayBase<D,Ix1>) -> MaxHeap<F,R> {
		let mut heap = MaxHeap::new();
		if ids.is_some() {
			let ids = unsafe{ids.unwrap_unchecked()};
			random_unique_uint::<R>(ids.len(), k).iter().for_each(|&v|
				heap.push(self.half_indexed_distance(ids[unsafe{v.to_usize().unwrap_unchecked()}], q), v)
			);
		} else {
			random_unique_uint::<R>(self.n_rows(), k).iter().for_each(|&v|
				heap.push(self.half_indexed_distance(v, q), v)
			);
		}
		heap
	}
	/// Executes a greedy search on the given layer for the approximate k nearest neighbors and returns the indices and distances to these neighbors.
	fn greedy_search_layer<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize, layer: usize) -> Result<(Array1<R>, Array1<F>), NoSuchLayerError> {
		assert!(layer < self.layer_count());
		/* Randomly initialized search heap */
		let heap = self._random_init_maxheap(self.get_global_layer_ids(0), max_heap_size, q);
		/* Search the specified layer */
		let heap = self.greedy_search_layer_with_heap(q, heap, max_heap_size, layer)?;
		/* Extract the k nearest neighbors */
		let mut ids = Array1::from_elem(k_neighbors, R::zero());
		let mut dists = Array1::from_elem(k_neighbors, F::zero());
		heap.into_iter().collect::<Vec<_>>().into_iter().rev()
		.enumerate().filter(|(i,_)| *i < k_neighbors)
		.for_each(|(i,(d, v))| { ids[i] = v; dists[i] = d; });
		/* Return the result */
		Ok((ids, dists))
	}
	/// Executes a greedy search on the hierarchy with a maximum heap size and returns the heap containing the results.
	fn greedy_search<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, k_neighbors: usize, max_heap_size: usize) -> (Array1<R>, Array1<F>) {
		/* Randomly initialized search heap */
		let heap = self._random_init_maxheap(self.get_global_layer_ids(0), max_heap_size, q);
		/* Search all layers graph */
		let heap = self.greedy_search_with_heap(q, heap, max_heap_size);
		/* Extract the k nearest neighbors */
		let mut ids = Array1::from_elem(k_neighbors, R::zero());
		let mut dists = Array1::from_elem(k_neighbors, F::zero());
		heap.into_iter().collect::<Vec<_>>().into_iter().rev()
		.enumerate().filter(|(i,_)| *i < k_neighbors)
		.for_each(|(i,(d, v))| { ids[i] = v; dists[i] = d; });
		/* Return the result */
		(ids, dists)
	}
	fn greedy_search_batch<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix2>, k_neighbors: usize, max_heap_size: usize) -> (Array2<R>, Array2<F>) {
		let mut ids = Array2::from_elem((q.dim().0, k_neighbors), R::zero());
		let mut dists = Array2::from_elem((q.dim().0, k_neighbors), F::zero());
		q.axis_iter(ndarray::Axis(0)).enumerate().for_each(|(i, q)| {
			let (ids_i, dists_i) = self.greedy_search(&q, k_neighbors, max_heap_size);
			ids.index_axis_mut(ndarray::Axis(0), i).assign(&ids_i);
			dists.index_axis_mut(ndarray::Axis(0), i).assign(&dists_i);
		});
		(ids, dists)
	}
	/// Executes a greedy search on the given layer with a potentially pre-filled heap and returns the heap containing the results.
	fn greedy_search_layer_with_heap<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, heap: MaxHeap<F,R>, max_heap_size: usize, layer: usize) -> Result<MaxHeap<F,R>, NoSuchLayerError>;
	/// Executes a greedy search on the hierarchy with a potentially pre-filled heap and returns the heap containing the results.
	fn greedy_search_with_heap<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, mut heap: MaxHeap<F,R>, max_heap_size: usize) -> MaxHeap<F,R> {
		for layer in (0..self.layer_count()).rev() {
			/* Get heap from the current layer */
			heap = self.greedy_search_layer_with_heap(q, heap, max_heap_size, layer).unwrap();
			/* If it is not the last layer, assume that the layer has an ID map */
			if layer != self.layer_count()-1 {
				let idx_map = self.get_global_layer_ids(layer);
				if idx_map.is_some() {
					let idx_map = unsafe{idx_map.unwrap_unchecked()};
					/* Re-index entries in the heap with the local ID map */
					let mut new_heap = MaxHeap::new();
					heap.into_iter().for_each(|(d, v)| new_heap.push(d, idx_map[v.to_usize().unwrap()]));
					heap = new_heap;
				}
			}
		}
		heap
	}
}


pub struct GreedySingleGraphIndex<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> {
	_phantom: std::marker::PhantomData<(R,F)>,
	data: Mat,
	graph: G,
	distance: Dist,
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> GreedySingleGraphIndex<R, F, Dist, Mat, G> {
	pub fn new(data: Mat, graph: G, distance: Dist) -> Self {
		Self{
			_phantom: std::marker::PhantomData,
			data,
			graph,
			distance,
		}
	}
	pub fn n_edges(&self) -> usize { self.graph.n_edges() }
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> MatrixDataSource<F> for GreedySingleGraphIndex<R, F, Dist, Mat, G> {
	fn n_rows(&self) -> usize { self.data.n_rows() }
	fn n_cols(&self) -> usize { self.data.n_cols() }
	fn get_row(&self, i_row: usize) -> Array1<F> { self.data.get_row(i_row) }
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<F> { self.data.get_rows(i_rows) }
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<F> { self.data.get_rows_slice(i_row_from, i_row_to) }
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> RangeIndex<R,F,Dist> for GreedySingleGraphIndex<R, F, Dist, Mat, G> {
	fn range_query<D: Data<Elem=F>>(&self, _query: &ArrayBase<D,Ix1>, _range: F) -> (Array1<R>, Array1<F>) {
		panic!("Not implemented");
	}
	fn range_query_batch<D: Data<Elem=F>>(&self, _query: &ArrayBase<D,Ix2>, _range: F) -> (Vec<Array1<R>>, Vec<Array1<F>>) {
		panic!("Not implemented");
	}
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> KnnIndex<R,F,Dist> for GreedySingleGraphIndex<R, F, Dist, Mat, G> {
	fn knn_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D,Ix1>, k: usize) -> (Array1<R>, Array1<F>) {
		self.greedy_search(query, k, 2*k)
	}
	fn knn_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D,Ix2>, k: usize) -> (Array2<R>, Array2<F>) {
		let mut ids = Array2::from_elem((query.dim().0, k), R::zero());
		let mut dists = Array2::from_elem((query.dim().0, k), F::zero());
		query.axis_iter(ndarray::Axis(0)).enumerate().for_each(|(i, q)| {
			let (ids_i, dists_i) = self.greedy_search(&q, k, 2*k);
			ids.index_axis_mut(ndarray::Axis(0), i).assign(&ids_i);
			dists.index_axis_mut(ndarray::Axis(0), i).assign(&dists_i);
		});
		(ids, dists)
	}
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> IndexedDistance<R,F,Dist> for GreedySingleGraphIndex<R, F, Dist, Mat, G> {
	fn distance<D1: Data<Elem=F>, D2: Data<Elem=F>>(&self, i: &ArrayBase<D1,Ix1>, j: &ArrayBase<D2,Ix1>) -> F {
		self.distance.dist(i, j)
	}
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> GraphIndex<R, F, Dist> for GreedySingleGraphIndex<R, F, Dist, Mat, G> {
	fn layer_count(&self) -> usize { 1 }
	fn get_layer(&self, layer: usize) -> Result<&dyn Graph<R>, NoSuchLayerError> { if layer==0 {Ok(&self.graph)} else {Err(NoSuchLayerError)} }
	fn get_global_layer_ids(&self, _layer: usize) -> Option<&Vec<R>> { None }
	fn get_local_layer_ids(&self, _layer: usize) -> Option<&Vec<R>> { None }
	fn greedy_search_layer_with_heap<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, mut heap: MaxHeap<F,R>, max_heap_size: usize, _layer: usize) -> Result<MaxHeap<F,R>, NoSuchLayerError> {
		let mut visited_set: HashSet<R> = HashSet::new();
		let mut frontier = MinHeap::new();
		heap.iter().map(|p: &MaxHeapPair<F,R>| (*p.key_ref(), *p.value_ref())).for_each(|(d,v)| {
			frontier.push(d, v);
			visited_set.insert(v);
		});
		while let Some((d, v)) = frontier.pop() {
			if d > heap.peek().unwrap().0 { break; }
			// if visited_set.contains(&v) { continue; }
			for &i in &self.graph.neighbors(v) {
				if !visited_set.contains(&i) {
					let neighbor_dist = self.half_indexed_distance(i, q);
					if heap.size() < max_heap_size {
						heap.push(neighbor_dist, i);
					} else if heap.peek().unwrap().0 > neighbor_dist {
						heap.pop();
						heap.push(neighbor_dist, i);
					}
					frontier.push(neighbor_dist, i);
					visited_set.insert(i);
				}
			}
		}
		Ok(heap)
	}
}

pub struct GreedyCappedSingleGraphIndex<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> {
	_phantom: std::marker::PhantomData<(R,F)>,
	data: Mat,
	graph: G,
	distance: Dist,
	max_frontier_size: usize,
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
	pub fn new(data: Mat, graph: G, distance: Dist, max_frontier_size: usize) -> Self {
		Self{
			_phantom: std::marker::PhantomData,
			data,
			graph,
			distance,
			max_frontier_size,
		}
	}
	pub fn n_edges(&self) -> usize { self.graph.n_edges() }
	pub fn max_frontier_size(&self) -> usize { self.max_frontier_size }
	pub fn set_max_frontier_size(&mut self, max_frontier_size: usize) { self.max_frontier_size = max_frontier_size; }
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> MatrixDataSource<F> for GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
	fn n_rows(&self) -> usize { self.data.n_rows() }
	fn n_cols(&self) -> usize { self.data.n_cols() }
	fn get_row(&self, i_row: usize) -> Array1<F> { self.data.get_row(i_row) }
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<F> { self.data.get_rows(i_rows) }
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<F> { self.data.get_rows_slice(i_row_from, i_row_to) }
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> RangeIndex<R,F,Dist> for GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
	fn range_query<D: Data<Elem=F>>(&self, _query: &ArrayBase<D,Ix1>, _range: F) -> (Array1<R>, Array1<F>) {
		panic!("Not implemented");
	}
	fn range_query_batch<D: Data<Elem=F>>(&self, _query: &ArrayBase<D,Ix2>, _range: F) -> (Vec<Array1<R>>, Vec<Array1<F>>) {
		panic!("Not implemented");
	}
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> KnnIndex<R,F,Dist> for GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
	fn knn_query<D: Data<Elem=F>>(&self, query: &ArrayBase<D,Ix1>, k: usize) -> (Array1<R>, Array1<F>) {
		self.greedy_search(query, k, 2*k)
	}
	fn knn_query_batch<D: Data<Elem=F>>(&self, query: &ArrayBase<D,Ix2>, k: usize) -> (Array2<R>, Array2<F>) {
		let mut ids = Array2::from_elem((query.dim().0, k), R::zero());
		let mut dists = Array2::from_elem((query.dim().0, k), F::zero());
		query.axis_iter(ndarray::Axis(0)).enumerate().for_each(|(i, q)| {
			let (ids_i, dists_i) = self.greedy_search(&q, k, 2*k);
			ids.index_axis_mut(ndarray::Axis(0), i).assign(&ids_i);
			dists.index_axis_mut(ndarray::Axis(0), i).assign(&dists_i);
		});
		(ids, dists)
	}
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> IndexedDistance<R,F,Dist> for GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
	fn distance<D1: Data<Elem=F>, D2: Data<Elem=F>>(&self, i: &ArrayBase<D1,Ix1>, j: &ArrayBase<D2,Ix1>) -> F {
		self.distance.dist(i, j)
	}
}
impl<R: UnsignedInteger, F: Float, Dist: Distance<F>, Mat: MatrixDataSource<F>, G: Graph<R>> GraphIndex<R, F, Dist> for GreedyCappedSingleGraphIndex<R, F, Dist, Mat, G> {
	fn layer_count(&self) -> usize { 1 }
	fn get_layer(&self, layer: usize) -> Result<&dyn Graph<R>, NoSuchLayerError> { if layer==0 {Ok(&self.graph)} else {Err(NoSuchLayerError)} }
	fn get_global_layer_ids(&self, _layer: usize) -> Option<&Vec<R>> { None }
	fn get_local_layer_ids(&self, _layer: usize) -> Option<&Vec<R>> { None }
	fn greedy_search_layer_with_heap<D: Data<Elem=F>>(&self, q: &ArrayBase<D,Ix1>, mut heap: MaxHeap<F,R>, max_heap_size: usize, _layer: usize) -> Result<MaxHeap<F,R>, NoSuchLayerError> {
		let mut visited_set: HashSet<R> = HashSet::new();
		let mut frontier = DualHeap::with_capacity(self.max_frontier_size);
		heap.iter().map(|p: &MaxHeapPair<F,R>| (*p.key_ref(), *p.value_ref())).for_each(|(d,v)| {
			if frontier.size() < self.max_frontier_size {
				frontier.push(d, v);
			} else {
				frontier.push_pop::<false>(d, v);
			}
			visited_set.insert(v);
		});
		while let Some((d, v)) = frontier.pop::<true>() {
			if d > heap.peek().unwrap().0 { break; }
			// if visited_set.contains(&v) { continue; }
			for &i in &self.graph.neighbors(v) {
				if !visited_set.contains(&i) {
					let neighbor_dist = self.half_indexed_distance(i, q);
					if heap.size() < max_heap_size {
						heap.push(neighbor_dist, i);
					} else if heap.peek().unwrap().0 > neighbor_dist {
						heap.pop();
						heap.push(neighbor_dist, i);
					}
					if frontier.size() < self.max_frontier_size {
						frontier.push(neighbor_dist, i);
					} else {
						frontier.push_pop::<false>(neighbor_dist, i);
					}
					visited_set.insert(i);
				}
			}
		}
		Ok(heap)
	}
}

