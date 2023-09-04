use std::{ops::Range, fmt::Debug, collections::VecDeque};

#[derive(Debug, Clone)]
pub struct Bounds {
    pub x: Range<f32>,
    pub y: Range<f32>
}

impl Bounds {
    pub fn overlaps(&self, other: &Bounds) -> bool {
        // TODO: I feel like this can be made more efficient
        // ... although maybe compiler handles that
        (self.x.contains(&other.x.start) || other.x.contains(&self.x.start)) && 
        (self.y.contains(&other.y.start) || other.y.contains(&self.y.start))
    }

    pub fn contains(&self, pos: (f32, f32)) -> bool {
        self.x.contains(&pos.0) && self.y.contains(&pos.1)
    }

    pub fn center(&self) -> (f32, f32) {
        (
            (self.x.start + self.x.end) / 2.0,
            (self.y.start + self.y.end) / 2.0
        )
    }

    pub fn overlaps_circle(&self, center: (f32, f32), radius: f32) -> bool {
        let bcenter = self.center();
        let dirvec = (center.0 - bcenter.0, center.1 - bcenter.1);
        let dirmag_sq = dirvec.0 * dirvec.0 + dirvec.1 * dirvec.1;
        // is the bounds center contained within the circle
        // this is needed for the case where the bounds is a subset of the circle
        if dirmag_sq < radius * radius {
            return true;
        }
        let dirmag_inv = 1.0/dirmag_sq.sqrt();
        let closest_point = (dirvec.0 * dirmag_inv, dirvec.1 * dirmag_inv);
        return self.contains(closest_point)
    }

    fn new(x: Range<f32>, y: Range<f32>) -> Self {
        Bounds { x , y  }
    }

    fn unit() -> Self {
        Bounds { x: 0.0..1.0 , y: 0.0..1.0  }
    }
}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct Node<T> {
    pub pos: (f32, f32),
    pub item: T
}


impl<T> Debug for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("pos", &self.pos)
            // .field("item", &self.item)
            .finish()
    }
}

#[derive(Debug)]
enum QCell<T: Eq> {
    Term(Vec<Node<T>>),
    NonTerm([Box<QuadTree<T>>; 4]),
}

impl<T: Eq> QCell<T> {
    const LIMIT: usize = 64;
}


#[derive(Debug)]
pub struct QuadTree<T>
where 
    T: Eq
{
    bounds: Bounds,
    inner: QCell<T>
}

impl<T: Eq> QuadTree<T> {
    pub fn new() -> Self {
        Self {
            bounds: Bounds::unit(),
            inner: QCell::Term(Vec::with_capacity(QCell::<T>::LIMIT)),
        }
    }
    pub fn new_with_bounds(bounds: Bounds) -> Self {
        Self {
            bounds,
            inner: QCell::Term(Vec::with_capacity(QCell::<T>::LIMIT)),
        }
    }

    pub fn insert(&mut self, item: Node<T>) -> Result<(), Node<T>> {
        if !self.bounds.contains(item.pos){
            return Err(item);
        }
        match &mut self.inner {
            QCell::Term(v) if v.len() < QCell::<T>::LIMIT => {
                v.push(item)
            },
            QCell::Term(v) => {
                let v = std::mem::take(v);
                let divx = (self.bounds.x.start + self.bounds.x.end) / 2.0;
                let divy = (self.bounds.y.start + self.bounds.y.end) / 2.0;
                let ranges = [
                    Bounds::new(self.bounds.x.start..divx, self.bounds.y.start..divy),
                    Bounds::new(divx..self.bounds.x.end, self.bounds.y.start..divy),
                    Bounds::new(divx..self.bounds.x.end, divy..self.bounds.y.end),
                    Bounds::new(self.bounds.x.start..divx, divy..self.bounds.y.end)
                ];
                let mut new = [
                    Box::new(Self::new_with_bounds(ranges[0].clone())),
                    Box::new(Self::new_with_bounds(ranges[1].clone())),
                    Box::new(Self::new_with_bounds(ranges[2].clone())),
                    Box::new(Self::new_with_bounds(ranges[3].clone())),
                ];
                for transfer in v.into_iter().chain([item]) {
                    match (transfer.pos.0 < divx, transfer.pos.1 < divy) {
                        (true, true) => new[0].insert(transfer).expect("pre-existing item is valid"),
                        (false, true) => new[1].insert(transfer).expect("pre-existing item is valid"),
                        (false, false) => new[2].insert(transfer).expect("pre-existing item is valid"),
                        (true, false) => new[3].insert(transfer).expect("pre-existing item is valid"),
                    }
                }
                self.inner = QCell::NonTerm(new)
            },
            QCell::NonTerm(a) => {
                let divx = (self.bounds.x.start + self.bounds.x.end) / 2.0;
                let divy = (self.bounds.y.start + self.bounds.y.end) / 2.0;
                match (item.pos.0 < divx, item.pos.1 < divy) {
                    (true, true) => a[0].insert(item).expect("already checked validity"),
                    (false, true) => a[1].insert(item).expect("already checked validity"),
                    (false, false) => a[2].insert(item).expect("already checked validity"),
                    (true, false) => a[3].insert(item).expect("already checked validity"),
                }
            }
        }
        Ok(())
    }


    pub fn query_rect(&self, bounds: &'_ Bounds) -> impl Iterator<Item = &Node<T>> {
        QuadTreeIterator {
            curr: None,
            queue: vec![self].into(),
            bounds: bounds.clone()
        }
    }
}

pub struct QuadTreeIterator<'a, T>
where
    T: Eq
{
    curr: Option<std::slice::Iter<'a, Node<T>>>,
    queue: VecDeque<&'a QuadTree<T>>,
    bounds: Bounds
}

impl<'a, T> Iterator for QuadTreeIterator<'a, T>
    where 
    T: Eq
{
    type Item = &'a Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while let Some(ret) = self.curr.as_mut().and_then(|it| it.next()) {
                if self.bounds.contains(ret.pos) {
                    return Some(ret);
                }
            }
            let next = self.queue.pop_front()?;
            match &next.inner {
                QCell::Term(v) => self.curr = Some(v.iter()),
                QCell::NonTerm(a) => self.queue.extend(a.iter().filter(|q| q.bounds.overlaps(&self.bounds)).map(|q| q.as_ref())),
            }
        }
    }
}

impl<A: Eq> Extend<Node<A>> for QuadTree<A> {
    fn extend<T: IntoIterator<Item = Node<A>>>(&mut self, iter: T) {
        for i in iter {
            self.insert(i).expect("out of bounds insertion");
        }
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    fn diag_node(x: f32) -> Node<()> {
        Node { pos: (x, x), item: () }
    }

    fn grid(dim: usize) -> Vec<Node<()>> {
        let mut ret = Vec::with_capacity(dim * dim);
        for x in 0..dim {
            let x = x as f32 / dim as f32;
            for y in 0..dim {
                let y = y as f32 / dim as f32;
                ret.push(Node { pos: (x,y), item: ()});
            }
        }
        ret
    }

    fn bounds_check(q: &mut QuadTree<()>, v: Vec<Node<()>>, b: Bounds) {
        let mut expected: Vec<_> = v.iter().filter(|n| b.contains(n.pos)).map(std::clone::Clone::clone).collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        q.extend(v);
        let mut actual = Vec::from_iter(q.query_rect(&b).map(std::clone::Clone::clone));
        actual.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(expected, actual);
    }

    #[test]
    fn bounds() {
        let b = Bounds::unit();
        assert!(b.contains((0.0, 0.0)));
        assert!(!b.contains((1.0, 0.0)));
        assert!(!b.contains((1.0, 1.0)));
        assert!(!b.contains((0.0, 1.0)));
        assert!(b.contains((0.5, 0.5)));
    }

    #[test]
    fn small_count() {
        let mut q = QuadTree::new();
        for x in 0..8 {
            q.insert(diag_node(x as f32 / 8 as f32)).unwrap()
        }
    }

    #[test]
    fn large_count() {
        let mut q = QuadTree::new();
        for x in 0..1024 {
            let x = x % 256;
            q.insert(diag_node(x as f32 / 256 as f32)).unwrap()
        }
        // dbg!(&q);
        // panic!()
    }

    #[test]
    // #[ignore= "overflow"]
    fn larger_count() {
        let mut q = QuadTree::new();
        for x in 0..(1 << 15) {
            let x = x % 256;
            q.insert(diag_node(x as f32 / 256 as f32)).unwrap()
        }
    }

    #[test]
    fn grid_bounds_tiny() {
        let b = Bounds::new(0.2123..0.702, 0.7..0.931);
        let mut q = QuadTree::new();
        let v = grid(4);
        bounds_check(&mut q, v, b);
    }

    #[test]
    fn grid_bounds_small() {
        let b = Bounds::new(0.2123..0.702, 0.7..0.931);
        let mut q = QuadTree::new();
        let v = grid(8);
        bounds_check(&mut q, v, b);
    }

    #[test]
    fn grid_bounds_large() {
        let b = Bounds::new(0.2123..0.702, 0.7..0.931);
        let mut q = QuadTree::new();
        let v = grid(32);
        bounds_check(&mut q, v, b);
    }

    #[test]
    fn grid_bounds_uneven() {
        let b = Bounds::new(0.2123..0.702, 0.7..0.931);
        let mut q = QuadTree::new();
        let mut v = grid(32);
        v.extend(grid(16).into_iter().map(|n| Node { pos: (n.pos.0 / 3.0 + 0.5, n.pos.1 / 4.0 + 0.7), item: () }));
        bounds_check(&mut q, v, b);
    }
}
