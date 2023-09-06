use quadtree::*;


fn main() {
    let mut qt = QuadTree::new();
    let mut rng = fastrand::Rng::new();

    qt.extend({
        (0..100_000_000).map(|n| Node::from_pos((rng.f32(), rng.f32()), n ))
    });

    let start = std::time::Instant::now();
    let v: Vec<_> = qt.query_rect(&Bounds::new(0.2..0.23, 0.74..0.76)).collect();
    println!("elapsed = {}", start.elapsed().as_micros());
    println!("res = {:?}", &v[..3]);
}
