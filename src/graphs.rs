use std::collections::{HashMap, HashSet};

use crate::{bit_vectors::{BitVector, BitVectorMut}, heaps::MinHeap, types::{Float, UnsignedInteger}};


pub trait Graph<R: UnsignedInteger> {
	fn reserve(&mut self, n_vertices: usize);
	fn n_vertices(&self) -> usize;
	fn n_edges(&self) -> usize;
	fn neighbors(&self, vertex: R) -> Vec<R>;
	fn add_node(&mut self);
	fn add_node_with_capacity(&mut self, capacity: usize);
	fn add_edge(&mut self, vertex1: R, vertex2: R);
	fn find_edge(&self, vertex1: R, vertex2: R) -> Option<usize> {
		self.neighbors(vertex1).iter().position(|&v| v == vertex2)
	}
	fn remove_edge(&mut self, vertex1: R, vertex2: R) {
		let index = self.find_edge(vertex1, vertex2);
		if index.is_some() {
			self.remove_edge_by_index(vertex1, unsafe{index.unwrap_unchecked()});
		}
	}
	fn remove_edge_by_index(&mut self, vertex: R, index: usize);
	fn remove_edges_chunk(&mut self, vertex1: R, vertices2: &Vec<R>) {
		let mut remove_indices = vertices2.iter()
		.map(|&v| self.find_edge(vertex1, v))
		.filter(|v| v.is_some())
		.map(|v| unsafe{v.unwrap_unchecked()})
		.collect::<Vec<_>>();
		remove_indices.sort();
		remove_indices.into_iter().rev().for_each(|i| self.remove_edge_by_index(vertex1, i));
	}
	fn add_edges_chunk(&mut self, vertex1: R, vertices2: &Vec<R>) {
		vertices2.iter().for_each(|&v| self.add_edge(vertex1, v));
	}
	fn ego_graph_nodes_hashset(&self, vertex: R, radius: usize) -> Vec<R> {
		let mut visited = HashSet::new();
		let mut work = vec![vertex];
		let mut visited_ids = vec![vertex];
		visited.insert(vertex);
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				for &n in &self.neighbors(v) {
					if !visited.contains(&n) {
						visited.insert(n);
						next_work.push(n);
						visited_ids.push(n);
					}
				}
			}
			work = next_work;
		}
		visited_ids
	}
	fn ego_graph_nodes_bitvec(&self, vertex: R, radius: usize) -> Vec<R> {
		let mut visited = vec![0u64; (self.n_vertices()+63)/64];
		let mut work = vec![vertex];
		let mut visited_ids = vec![vertex];
		visited.set_bit_unchecked(vertex.to_usize().unwrap(), true);
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				for &n in &self.neighbors(v) {
					if !visited.get_bit_unchecked(n.to_usize().unwrap()) {
						visited.set_bit_unchecked(n.to_usize().unwrap(), true);
						next_work.push(n);
						visited_ids.push(n);
					}
				}
			}
			work = next_work;
		}
		visited_ids
	}
	fn as_dir_lol_graph(&self) -> DirLoLGraph<R> {
		let mut ret = DirLoLGraph::new();
		(0..self.n_vertices()).for_each(|_| ret.add_node());
		(0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).for_each(|i| {
			self.neighbors(i).into_iter().for_each(|j| ret.add_edge(i, j));
		});
		ret
	}
	fn as_undir_lol_graph(&self) -> UndirLoLGraph<R> {
		let mut ret: UndirLoLGraph<R> = UndirLoLGraph::new();
		(0..self.n_vertices()).for_each(|_| ret.add_node());
		if self.n_edges() == 0 { return ret; }
		/* Get edges in ascending node order */
		let mut edges = (0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).flat_map(|i| {
			self.neighbors(i).into_iter().map(move |j| (i.min(j),i.max(j)))
		}).collect::<Vec<_>>();
		/* Add edges if not equal to the previous one (unique edges) */
		edges.sort();
		ret.add_edge(edges[0].0, edges[0].1);
		(1..edges.len())
		.filter(|&i| edges[i] != edges[i-1])
		.for_each(|i| ret.add_edge(edges[i].0, edges[i].1));
		ret
	}
	fn as_viewable_adj_graph(&self) -> Option<&dyn ViewableAdjGraph<R>> { None }
	fn as_vec_viewable_adj_graph(&self) -> Option<&dyn VecViewableAdjGraph<R>> { None }
}
pub trait WeightedGraph<R: UnsignedInteger, F: Float>: Graph<R> {
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F;
	fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F);
	fn add_edges_with_weight_chunk(&mut self, vertex1: R, vertices2: &Vec<R>, weights: &Vec<F>) {
		vertices2.iter().zip(weights.iter())
		.for_each(|(&v,&w)| self.add_edge_with_weight(vertex1, v, w));
	}
	fn add_edges_with_zipped_weight_chunk(&mut self, vertex1: R, vertices2: &Vec<(F,R)>) {
		vertices2.iter()
		.for_each(|&(w,v)| self.add_edge_with_weight(vertex1, v, w));
	}
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>);
	fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F,R)>;
	fn weighted_ego_graph_nodes(&self, vertex: R, radius: usize) -> (Vec<F>, Vec<R>) {
		let mut visited = HashMap::new();
		let mut work = vec![vertex];
		visited.insert(vertex, F::zero());
		for _ in 0..radius {
			let mut next_work = Vec::new();
			for &v in &work {
				let v_dist = visited[&v];
				let (neighbors, weights) = self.neighbors_with_weights(v);
				for (&w, &n) in neighbors.iter().zip(weights.iter()) {
					if !visited.contains_key(&n) || visited[&n] > v_dist + w {
						visited.insert(n, v_dist + w);
						next_work.push(n);
					}
				}
			}
			work = next_work;
		}
		let mut heap = MinHeap::new();
		for (&v, &d) in &visited {
			heap.push(d, v);
		}
		let mut visited_ids = Vec::with_capacity(heap.size());
		let mut visited_dists = Vec::with_capacity(heap.size());
		for (d, v) in heap.into_iter() {
			visited_ids.push(v);
			visited_dists.push(d);
		}
		(visited_dists, visited_ids)
	}
	fn as_weighted_dir_lol_graph(&self) -> WDirLoLGraph<R,F> {
		let mut ret = WDirLoLGraph::new();
		(0..self.n_vertices()).for_each(|_| ret.add_node());
		(0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).for_each(|i| {
			let (neighbors, weights) = self.neighbors_with_weights(i);
			neighbors.into_iter().zip(weights.into_iter()).for_each(|(d,j)| ret.add_edge_with_weight(i, j, d));
		});
		ret
	}
	fn as_weighted_undir_lol_graph(&self) -> WUndirLoLGraph<R,F> {
		let mut ret: WUndirLoLGraph<R,F> = WUndirLoLGraph::new();
		(0..self.n_vertices()).for_each(|_| ret.add_node());
		if self.n_edges() == 0 { return ret; }
		/* Get edges in ascending node order */
		let mut edges = (0..self.n_vertices()).map(|i| unsafe{R::from_usize(i).unwrap_unchecked()}).flat_map(|i| {
			let (neighbors, weights) = self.neighbors_with_weights(i);
			neighbors.into_iter().zip(weights.into_iter()).map(move |(d,j)| (i.min(j),i.max(j),d))
		}).collect::<Vec<_>>();
		/* Add edges if not equal to the previous one (unique edges) */
		/* If both directions are available with different edge weights, the behavior is undefined */
		edges.sort_by_key(|&(i,j,_)| (i,j));
		ret.add_edge(edges[0].0, edges[0].1);
		(1..edges.len())
		.filter(|&i| edges[i].0 != edges[i-1].0 || edges[i].1 != edges[i-1].1)
		.for_each(|i| ret.add_edge_with_weight(edges[i].0, edges[i].1, edges[i].2));
		ret
	}
	fn as_viewable_weighted_adj_graph(&self) -> Option<&dyn ViewableWeightedAdjGraph<R,F>> { None }
	fn as_vec_viewable_weighted_adj_graph(&self) -> Option<&dyn VecViewableWeightedAdjGraph<R,F>> { None }
}
pub trait ViewableAdjGraph<R: UnsignedInteger>: Graph<R> {
	fn view_neighbors(&self, vertex: R) -> &[R];
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [R];
}
pub trait VecViewableAdjGraph<R: UnsignedInteger>: Graph<R> {
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<R>;
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<R>;
}
pub trait ViewableWeightedAdjGraph<R: UnsignedInteger, F: Float>: WeightedGraph<R,F> {
	fn view_neighbors(&self, vertex: R) -> &[(F,R)];
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F,R)];
}
pub trait VecViewableWeightedAdjGraph<R: UnsignedInteger, F: Float>: WeightedGraph<R,F> {
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<(F,R)>;
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<(F,R)>;
}


pub struct DirLoLGraph<R: UnsignedInteger> {
	adjacency: Vec<Vec<R>>,
	n_edges: usize
}
impl<R: UnsignedInteger> DirLoLGraph<R> {
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
}
impl<R: UnsignedInteger> Graph<R> for DirLoLGraph<R> {
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	fn neighbors(&self, vertex: R) -> Vec<R> {
		self.adjacency[vertex.to_usize().unwrap()].clone()
	}
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	fn add_edge(&mut self, vertex1: R, vertex2: R) {
		self.adjacency[vertex1.to_usize().unwrap()].push(vertex2);
		self.n_edges += 1;
	}
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		self.adjacency[vertex.to_usize().unwrap()].swap_remove(index);
		self.n_edges -= 1;
	}
	fn as_viewable_adj_graph(&self) -> Option<&dyn ViewableAdjGraph<R>> {
		Some(self)
	}
	fn as_vec_viewable_adj_graph(&self) -> Option<&dyn VecViewableAdjGraph<R>> {
		Some(self)
	}
}
impl<R: UnsignedInteger> ViewableAdjGraph<R> for DirLoLGraph<R> {
	fn view_neighbors(&self, vertex: R) -> &[R] {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [R] {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
impl<R: UnsignedInteger> VecViewableAdjGraph<R> for DirLoLGraph<R> {
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<R> {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<R> {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
pub struct UndirLoLGraph<R: UnsignedInteger> {
	adjacency: Vec<Vec<R>>,
	n_edges: usize
}
impl<R: UnsignedInteger> UndirLoLGraph<R> {
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
}
impl<R: UnsignedInteger> Graph<R> for UndirLoLGraph<R> {
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	fn neighbors(&self, vertex: R) -> Vec<R> {
		self.adjacency[vertex.to_usize().unwrap()].clone()
	}
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	fn add_edge(&mut self, vertex1: R, vertex2: R) {
		self.adjacency[vertex1.to_usize().unwrap()].push(vertex2);
		self.adjacency[vertex2.to_usize().unwrap()].push(vertex1);
		self.n_edges += 1;
	}
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		self.adjacency[vertex.to_usize().unwrap()].swap_remove(index);
		let neighbor = self.adjacency[vertex.to_usize().unwrap()][index];
		let neighbor_index = self.adjacency[neighbor.to_usize().unwrap()].iter().position(|&v| v == vertex).unwrap();
		self.adjacency[neighbor.to_usize().unwrap()].swap_remove(neighbor_index);
		self.n_edges -= 1;
	}
	fn as_viewable_adj_graph(&self) -> Option<&dyn ViewableAdjGraph<R>> {
		Some(self)
	}
	fn as_vec_viewable_adj_graph(&self) -> Option<&dyn VecViewableAdjGraph<R>> {
		Some(self)
	}
}
impl<R: UnsignedInteger> ViewableAdjGraph<R> for UndirLoLGraph<R> {
	fn view_neighbors(&self, vertex: R) -> &[R] {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [R] {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
impl<R: UnsignedInteger> VecViewableAdjGraph<R> for UndirLoLGraph<R> {
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<R> {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<R> {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
pub struct WDirLoLGraph<R: UnsignedInteger, F: Float> {
	adjacency: Vec<Vec<(F,R)>>,
	n_edges: usize
}
impl<R: UnsignedInteger, F: Float> WDirLoLGraph<R,F> {
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
}
impl<R: UnsignedInteger, F: Float> Graph<R> for WDirLoLGraph<R,F> {
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	fn neighbors(&self, vertex: R) -> Vec<R> {
		self.adjacency[vertex.to_usize().unwrap()].iter().map(|&(_,v)| v).collect()
	}
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	fn add_edge(&mut self, _vertex1: R, _vertex2: R) {
		panic!("Cannot add edge without weight to a weighted graph");
	}
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		self.adjacency[vertex.to_usize().unwrap()].swap_remove(index);
		self.n_edges -= 1;
	}
}
impl<R: UnsignedInteger, F: Float> WeightedGraph<R,F> for WDirLoLGraph<R,F> {
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F {
		self.adjacency[vertex1.to_usize().unwrap()].iter().find(|&&(_,v)| v == vertex2).unwrap().0
	}
	fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F) {
		self.adjacency[vertex1.to_usize().unwrap()].push((weight, vertex2));
		self.n_edges += 1;
	}
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>) {
		let mut neighbors = Vec::new();
		let mut weights = Vec::new();
		for &(w,v) in &self.adjacency[vertex.to_usize().unwrap()] {
			neighbors.push(v);
			weights.push(w);
		}
		(weights, neighbors)
	}
	fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F,R)> {
		self.adjacency[vertex.to_usize().unwrap()].clone()
	}
	fn as_viewable_weighted_adj_graph(&self) -> Option<&dyn ViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
	fn as_vec_viewable_weighted_adj_graph(&self) -> Option<&dyn VecViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
}
impl<R: UnsignedInteger, F: Float> ViewableWeightedAdjGraph<R,F> for WDirLoLGraph<R,F> {
	fn view_neighbors(&self, vertex: R) -> &[(F,R)] {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F,R)] {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
impl<R: UnsignedInteger, F: Float> VecViewableWeightedAdjGraph<R,F> for WDirLoLGraph<R,F> {
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<(F,R)> {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<(F,R)> {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
pub struct WUndirLoLGraph<R: UnsignedInteger, F: Float> {
	adjacency: Vec<Vec<(F,R)>>,
	n_edges: usize
}
impl<R: UnsignedInteger, F: Float> WUndirLoLGraph<R,F> {
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
}
impl<R: UnsignedInteger, F: Float> Graph<R> for WUndirLoLGraph<R,F> {
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	fn neighbors(&self, vertex: R) -> Vec<R> {
		self.adjacency[vertex.to_usize().unwrap()].iter().map(|&(_,v)| v).collect()
	}
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	fn add_edge(&mut self, _vertex1: R, _vertex2: R) {
		panic!("Cannot add edge without weight to a weighted graph");
	}
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		self.adjacency[vertex.to_usize().unwrap()].swap_remove(index);
		let neighbor = self.adjacency[vertex.to_usize().unwrap()][index].0;
		let neighbor_index = self.adjacency[neighbor.to_usize().unwrap()].iter().position(|&v| v.1 == vertex).unwrap();
		self.adjacency[neighbor.to_usize().unwrap()].swap_remove(neighbor_index);
		self.n_edges -= 1;
	}
}
impl<R: UnsignedInteger, F: Float> WeightedGraph<R,F> for WUndirLoLGraph<R,F> {
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F {
		self.adjacency[vertex1.to_usize().unwrap()].iter().find(|&&(_,v)| v == vertex2).unwrap().0
	}
	fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F) {
		self.adjacency[vertex1.to_usize().unwrap()].push((weight, vertex2));
		self.adjacency[vertex2.to_usize().unwrap()].push((weight, vertex1));
		self.n_edges += 1;
	}
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>) {
		let mut neighbors = Vec::new();
		let mut weights = Vec::new();
		for &(w,v) in &self.adjacency[vertex.to_usize().unwrap()] {
			neighbors.push(v);
			weights.push(w);
		}
		(weights, neighbors)
	}
	fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F,R)> {
		self.adjacency[vertex.to_usize().unwrap()].clone()
	}
	fn as_viewable_weighted_adj_graph(&self) -> Option<&dyn ViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
	fn as_vec_viewable_weighted_adj_graph(&self) -> Option<&dyn VecViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
}
impl<R: UnsignedInteger, F: Float> ViewableWeightedAdjGraph<R,F> for WUndirLoLGraph<R,F> {
	fn view_neighbors(&self, vertex: R) -> &[(F,R)] {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F,R)] {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
impl<R: UnsignedInteger, F: Float> VecViewableWeightedAdjGraph<R,F> for WUndirLoLGraph<R,F> {
	fn view_neighbors_vec(&self, vertex: R) -> &Vec<(F,R)> {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	fn view_neighbors_vec_mut(&mut self, vertex: R) -> &mut Vec<(F,R)> {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}

