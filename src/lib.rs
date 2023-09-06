#![no_std]

struct MyStruct;

fn my_fn<F, T>(f: F)
where
    F: Fn(&T, &T) -> (),
{ }

fn rustc_panic() {

    // the unit struct does not work
    my_fn::<_, MyStruct>( |a, b| {
        a.partial_cmp(b)
    });

}
