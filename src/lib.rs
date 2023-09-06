use std::{collections::VecDeque, fmt::Debug, ops::Range};

type Index = u16;

#[derive(Debug, Clone)]
pub struct Bounds {
    pub x: Range<f32>,
    pub y: Range<f32>,
}

impl Bounds {
    pub fn overlaps(&self, other: &Bounds) -> bool {
        // TODO: I feel like this can be made more efficient
        // ... although maybe compiler handles that
        (self.x.contains(&other.x.start) || other.x.contains(&self.x.start))
            && (self.y.contains(&other.y.start) || other.y.contains(&self.y.start))
    }

    #[inline(always)]
    pub fn contains(&self, pos: (f32, f32)) -> bool {
        self.x.contains(&pos.0) && self.y.contains(&pos.1)
    }

    pub fn center(&self) -> (f32, f32) {
        (
            (self.x.start + self.x.end) / 2.0,
            (self.y.start + self.y.end) / 2.0,
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
        let dirmag_inv = 1.0 / dirmag_sq.sqrt();
        let closest_point = (dirvec.0 * dirmag_inv, dirvec.1 * dirmag_inv);
        self.contains(closest_point)
    }

    pub fn new(x: Range<f32>, y: Range<f32>) -> Self {
        Bounds { x, y }
    }

    fn is_subunit(&self) -> bool {
        self.x.start >= 0.0 && self.y.start >= 0.0 && self.x.end < 1.0 && self.y.end < 1.0
    }

    pub fn unit() -> Self {
        Bounds {
            x: 0.0..1.0,
            y: 0.0..1.0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct CellBounds {
    xl: Index,
    xh: Index,
    yl: Index,
    yh: Index,
}

impl CellBounds {
    const fn new() -> Self {
        CellBounds {
            xl: 0,
            xh: Index::MAX,
            yl: 0,
            yh: Index::MAX,
        }
    }

    const fn contains(&self, pos: CellPos, depth: u32) -> bool {
        assert!(depth < Index::BITS);
        let spx = pos.x >> depth;
        let spy = pos.y >> depth;
        self.xl >> depth <= spx &&
        self.xh >> depth >= spx &&
        self.yl >> depth <= spy &&
        self.yh >> depth >= spy
    }

    const fn intersects(&self, other: CellBounds, depth: u32) -> bool {
        assert!(depth < Index::BITS);
        (
            (self.xl >> depth <= other.xl >> depth && other.xl >> depth <= self.xh >> depth) ||
            (other.xl >> depth <= self.xl >> depth && self.xl >> depth <= other.xh >> depth)
        ) 
        &&
        (
            (self.yl >> depth <= other.yl >> depth && other.yl >> depth <= self.yh >> depth) ||
            (other.yl >> depth <= self.yl >> depth && self.yl >> depth <= other.yh >> depth)
        )
    }
    
    fn from_bounds(bounds: &Bounds) -> Self {
        assert!(bounds.is_subunit());
        let xh = {
            if bounds.x.start >= 1.0 {
                Index::MAX
            } else {
                (bounds.x.start * Index::MAX as f32) as Index
            }
        };
        let yh = {
            if bounds.x.start >= 1.0 {
                Index::MAX
            } else {
                (bounds.x.start * Index::MAX as f32) as Index
            }
        };

        CellBounds {
            xl: (bounds.x.start * Index::MAX as f32) as Index,
            xh,
            yl: (bounds.y.start * Index::MAX as f32) as Index,
            yh
        }
    }
}

#[derive(Clone)]
pub struct Node<T> {
    pos: CellPos,
    pub item: T,
}

impl<T: Debug> Debug for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.item.fmt(f)
    }
}

impl<T> Node<T> {
    pub fn from_pos(pos: (f32, f32), item: T) -> Self {
        Self {
            pos: CellPos::from_f32(pos),
            item,
        }
    }
}

#[derive(Debug)]
enum QCell<T> {
    Term(Vec<Node<T>>),
    NonTerm([Box<QuadTree<T>>; 4]),
}

impl<T> QCell<T> {
    const LIMIT: usize = 128;
}

#[derive(Clone, Copy)]
struct CellPos {
    x: Index,
    y: Index
}

impl CellPos {
    fn from_f32(pos: (f32, f32)) -> Self {
        if pos.0 < 0.0 || pos.0 >= 1.0 || pos.1 < 0.0 || pos.1 >= 1.0 {
            panic!("position must be in the range 0.0..1.0, given is {pos:?}");
        }

        CellPos {
            x: (pos.0 * Index::MAX as f32) as Index,
            y: (pos.1 * Index::MAX as f32) as Index}
    }

    fn new() -> Self {
        CellPos { x: 0, y: 0 }
    }

    /// depth is such that the lowest is 0
    #[inline(always)]
    const fn contains(&self, depth: u32, other: &CellPos) -> bool {
        assert!(depth < Index::BITS);
        self.x >> depth == other.x >> depth && self.y >> depth == other.y >> depth
    }

    #[inline(always)]
    const fn contains_split(&self, depth: u32, other: &CellPos) -> (bool, bool) {
        assert!(depth < Index::BITS);
        (self.x >> depth == other.x >> depth, self.y >> depth == other.y >> depth)
    }

    const fn split(&self, from_depth: u32) -> [CellPos; 4] {
        assert!(from_depth > 0);
        debug_assert!(self.x & ((1 << from_depth) - 1) == 0);
        debug_assert!(self.y & ((1 << from_depth) - 1) == 0);
        let depth = from_depth - 1;
        let b = 1 << depth;
        [
            CellPos { x: self.x    , y: self.y     },
            CellPos { x: self.x | b, y: self.y     },
            CellPos { x: self.x | b, y: self.y | b },
            CellPos { x: self.x    , y: self.y | b },
        ]
        
    }
}


impl Debug for CellPos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CellPos: {{ {:b}, {:b} }}", self.x, self.y)
    }
}

#[derive(Debug)]
pub struct QuadTree<T>
{
    pos: CellPos,
    inner: QCell<T>,
    weight: usize,
    depth: u32,
}

impl<T> QuadTree<T> {
    const MAX_DEPTH: u32 = Index::BITS;

    pub fn new() -> Self {
        Self {
            pos: CellPos::new(),
            inner: QCell::Term(Vec::with_capacity(QCell::<T>::LIMIT)),
            weight: 0,
            depth: Self::MAX_DEPTH,
        }
    }

    fn new_with_depth(pos: CellPos, depth: u32) -> Self {
        assert!(depth < Index::BITS);
        Self {
            pos,
            inner: QCell::Term(Vec::with_capacity(QCell::<T>::LIMIT)),
            weight: 0,
            depth,
        }
    }

    fn may_contain(&self, pos: &CellPos) -> bool {
        self.pos.contains(self.depth, &pos)
    }

    fn insert_inner(&mut self, node: Node<T>) {
        match &mut self.inner {
            QCell::Term(v) if v.len() < QCell::<T>::LIMIT || self.depth >= Self::MAX_DEPTH => {
                v.push(node)
            }
            QCell::Term(v) => {
                let v = std::mem::take(v);
                let ranges = self.pos.split(self.depth);
                let new_depth = self.depth - 1;
                let mut new = [
                    Box::new(Self::new_with_depth(ranges[0], new_depth)),
                    Box::new(Self::new_with_depth(ranges[1], new_depth)),
                    Box::new(Self::new_with_depth(ranges[2], new_depth)),
                    Box::new(Self::new_with_depth(ranges[3], new_depth)),
                ];
                for transfer in v.into_iter().chain([node]) {
                    match self.pos.contains_split(new_depth, &transfer.pos) {
                        (true, true) => new[0]
                            .insert_inner(transfer),
                        (false, true) => new[1]
                            .insert_inner(transfer),
                        (false, false) => new[2]
                            .insert_inner(transfer),
                        (true, false) => new[3]
                            .insert_inner(transfer),
                    }
                }
                self.inner = QCell::NonTerm(new)
            }
            QCell::NonTerm(a) => {
                match self.pos.contains_split(self.depth, &node.pos) {
                    (true, true) => a[0]
                        .insert_inner(node),
                    (false, true) => a[1]
                        .insert_inner(node),
                    (false, false) => a[2]
                        .insert_inner(node),
                    (true, false) => a[3]
                        .insert_inner(node),
                }
            }
        }
        self.weight += 1;
    }

    pub fn insert(&mut self, pos: (f32, f32), item: T) -> Result<(), T> {
        let pos = CellPos::from_f32(pos);
        debug_assert!(self.may_contain(&pos));
        let node = Node { pos, item };
        self.insert_inner(node);
        Ok(())
    }

    fn maybe_absorb(&mut self) {
        if self.weight < QCell::<T>::LIMIT / 2 {
            return;
        }
        let QCell::NonTerm(_) = self.inner else { return };

        let old = std::mem::replace(
            &mut self.inner,
            QCell::Term(Vec::with_capacity(QCell::<T>::LIMIT)),
        );
        let QCell::NonTerm(old) = old else { panic!("just checked") };
        let QCell::Term(new) = &mut self.inner else { panic!("just checked") };
        new.extend(old.into_iter().map(|q| q.into_iter()).flatten());
    }

    pub fn retain_in_area_inner<F>(&mut self, bounds: CellBounds, func: &F) -> usize
    where
    F: Fn(&T) -> bool
    {
        let mut cnt;
        match &mut self.inner {
            QCell::Term(v) => {
                let start = v.len();
                v.retain(|n| func(&n.item));
                cnt = start - v.len();
            }
            QCell::NonTerm(a) => {
                cnt = 0;
                for q in a {
                    if bounds.contains(q.pos, q.depth) {
                        cnt += q.retain_in_area_inner(bounds, func);
                    }
                }
                if cnt > 0 {
                    self.maybe_absorb();
                }
            }
        }
        self.weight -= cnt;
        cnt
    }

    /// retain, but only look in a specific area. This area isn't exact, and the retention function
    /// will be called on every item in every cell that bounds intersects
    pub fn retain_in_area<F>(&mut self, bounds: &Bounds, func: &F) -> usize
    where
    F: Fn(&T) -> bool
    {
        assert!(bounds.is_subunit());
        self.retain_in_area_inner(CellBounds::from_bounds(bounds), func)
    }

    pub fn retain<F>(&mut self, func: &F) -> usize
    where
    F: Fn(&T) -> bool
    {
        let mut cnt;
        match &mut self.inner {
            QCell::Term(v) => {
                let start = v.len();
                v.retain(|n| func(&n.item));
                cnt = start - v.len();
            }
            QCell::NonTerm(a) => {
                cnt = 0;
                for q in a {
                    cnt += q.retain(func);
                }
                if cnt > 0 {
                    self.maybe_absorb();
                }
            }
        }
        self.weight -= cnt;
        cnt
    }

    pub fn query_rect(&self, bounds: &'_ Bounds) -> impl Iterator<Item = &T> {
        QuadTreeIterator {
            curr: None,
            queue: vec![self].into(),
            bounds: CellBounds::from_bounds(bounds),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        QuadTreeIterator {
            curr: None,
            queue: vec![self].into(),
            bounds: CellBounds::new(),
        }
    }

    pub fn get<U>(&self, item: &U) -> Option<&T>
    where
        T: PartialEq<U>,
    {
        self.iter().find(|&n| n == item)
    }

    pub fn get_bounded<U>(&self, item: &U, bounds: &Bounds) -> Option<&T>
    where
        T: PartialEq<U>,
    {
        self.query_rect(bounds).find(|&n| n == item)
    }

    pub fn remove<U>(&mut self, item: &U) -> Option<Node<T>>
    where
        T: PartialEq<U>,
    {
        match &mut self.inner {
            QCell::Term(v) => {
                let pos = v.iter().position(|i| &i.item == item)?;
                let ret = v.swap_remove(pos);
                self.weight -= 1;
                Some(ret)
            }
            QCell::NonTerm(a) => {
                let ret = a[0]
                    .remove(item)
                    .or_else(|| a[1].remove(item))
                    .or_else(|| a[2].remove(item))
                    .or_else(|| a[3].remove(item))?;
                self.weight -= 1;
                self.maybe_absorb();
                Some(ret)
            }
        }
    }

    pub fn len(&self) -> usize {
        self.weight
    }
}

pub struct QuadTreeIterator<'a, T>
{
    curr: Option<std::slice::Iter<'a, Node<T>>>,
    queue: VecDeque<&'a QuadTree<T>>,
    bounds: CellBounds,
}

impl<'a, T> Iterator for QuadTreeIterator<'a, T>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while let Some(ret) = self.curr.as_mut().and_then(|it| it.next()) {
                return Some(&ret.item);
            }
            let next = self.queue.pop_front()?;
            match &next.inner {
                QCell::Term(v) => self.curr = Some(v.iter()),
                QCell::NonTerm(a) => self.queue.extend(
                    a.iter()
                        .filter(|q| self.bounds.contains(q.pos, q.depth))
                        .map(|q| q.as_ref()),
                ),
            }
        }
    }
}

impl<T> IntoIterator for QuadTree<T> {
    type Item = Node<T>;

    type IntoIter = QuadTreeIntoIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        QuadTreeIntoIterator {
            curr: None,
            queue: vec![Box::new(self)].into(),
        }
    }
}

pub struct QuadTreeIntoIterator<T>
{
    curr: Option<<Vec<Node<T>> as IntoIterator>::IntoIter>,
    queue: VecDeque<Box<QuadTree<T>>>,
}

impl<T> Iterator for QuadTreeIntoIterator<T>
{
    type Item = Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ret) = self.curr.as_mut().and_then(|it| it.next()) {
                return Some(ret);
            }
            let next = self.queue.pop_front()?;
            match next.inner {
                QCell::Term(v) => self.curr = Some(v.into_iter()),
                QCell::NonTerm(a) => self.queue.extend(a.into_iter()),
            }
        }
    }
}

impl<A: Eq> Extend<Node<A>> for QuadTree<A> {
    fn extend<T: IntoIterator<Item = Node<A>>>(&mut self, iter: T) {
        for i in iter {
            self.insert_inner(i);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn grid(dim: usize) -> Vec<Node<(f32, f32)>> {
        let mut ret = Vec::with_capacity(dim * dim);
        for x in 0..dim {
            let x = x as f32 / dim as f32;
            for y in 0..dim {
                let y = y as f32 / dim as f32;
                ret.push(Node {
                    pos: CellPos::from_f32((x, y)),
                    item: (x, y),
                });
            }
        }
        ret
    }

    fn bounds_check(q: &mut QuadTree<(f32, f32)>, v: Vec<Node<(f32, f32)>>, b: Bounds) {
        let mut expected: Vec<_> = v
            .iter()
            .filter(|n| b.contains(n.item))
            .map(std::clone::Clone::clone)
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        q.extend(v);
        let mut actual = Vec::from_iter(q.query_rect(&b).map(std::clone::Clone::clone));
        actual.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(expected, actual);
    }

    fn diag(x: i32, scale: i32) -> (f32, f32) {
        assert!(x < scale);
        (x as f32 / scale as f32, x as f32 / scale as f32)
    }
    fn insert_diag(q: &mut QuadTree<(f32, f32)>, x: i32, scale: i32) {
        q.insert(diag(x, scale), diag(x, scale)).unwrap()
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
            q.insert((x as f32 / 8.0, x as f32 / 8.0), ()).unwrap()
        }
    }

    #[test]
    fn large_count() {
        let mut q = QuadTree::new();
        for x in 0..1024 {
            let x = x % 256;
            q.insert((x as f32 / 256.0, x as f32 / 256.0), ()).unwrap()
        }
        // dbg!(&q);
        // panic!()
    }

    #[test]
    // #[ignore= "overflow"]
    fn larger_count() {
        let mut q = QuadTree::new();
        for x in 0..(1 << 15) {
            let x = x % (1 << 12);
            q.insert(diag(x, 1 << 12), ()).unwrap()
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
        v.extend(grid(16).into_iter().map(|n| Node {
            pos: (n.pos.0 / 3.0 + 0.5, n.pos.1 / 4.0 + 0.7),
            item: (),
        }));
        bounds_check(&mut q, v, b);
    }

    #[test]
    fn grid_bounds_remove() {
        let b = Bounds::new(0.2123..0.702, 0.7..0.931);
        let mut q = QuadTree::new();
        let mut v = grid(32);
        q.extend(v.clone());
        let removed = q.remove_bounds(&b);
        assert_eq!(removed + q.len(), v.len());
        let mut actual: Vec<_> = q.into_iter().collect();
        actual.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.retain(|n| !b.contains(n.pos));
        let mut expected = v;
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(expected, actual);
    }

    #[test]
    fn grid_bounds_uneven_remove() {
        let b = Bounds::new(0.2123..0.702, 0.7..0.931);
        let mut q = QuadTree::new();
        let mut v = grid(32);
        v.extend(grid(16).into_iter().map(|n| Node {
            pos: (n.pos.0 / 3.0 + 0.5, n.pos.1 / 4.0 + 0.7),
            item: (),
        }));
        q.extend(v.clone());
        let removed = q.remove_bounds(&b);
        assert_eq!(removed + q.len(), v.len());
        let mut actual: Vec<_> = q.into_iter().collect();
        actual.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.retain(|n| !b.contains(n.pos));
        let mut expected = v;
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(expected, actual);
    }
}
