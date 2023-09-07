#![no_std]

struct MyStruct;

fn my_fn<F, T>(f: F)
where
    F: Fn(&T, &T) -> (),
{ }

fn my_fn2<T>(a: T, b: T) {
    a.partial_cmp(b)
} 

// note: rustc 1.71.0 (8ede3aae2 2023-07-12) running on x86_64-unknown-linux-gnu
fn rustc_panic() {

    MyStruct.partial_cmp(MyStruct);

    // the unit struct does not work
    my_fn::<_, MyStruct>( |a, b| {
        a.partial_cmp(b);
    });
}
