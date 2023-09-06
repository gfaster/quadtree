#![no_std]

pub struct MyStruct;

fn my_fn<F, T>(m: &mut [T], mut compare: F)
where
    F: FnMut(&T, &T) -> (),
{ }

/// offending function
fn rustc_panic() {
    // let _ = |a: MyStruct, b: MyStruct| a.partial_cmp(b).unwrap();

    my_fn(&mut [MyStruct], |a, b| {
        a.partial_cmp(b)
    });

}
