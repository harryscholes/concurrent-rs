use std::{
    ops::Deref,
    ptr::NonNull,
    sync::atomic::{
        fence, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
    },
};

#[derive(Debug, PartialEq)]
pub struct Arc<T> {
    inner: NonNull<Inner<T>>,
}

struct Inner<T> {
    data: Option<T>,
    strong_count: AtomicUsize,
    weak_count: AtomicUsize,
}

impl<T> Arc<T> {
    pub fn new(data: T) -> Self {
        Self {
            inner: NonNull::from(Box::leak(Box::new(Inner {
                data: Some(data),
                strong_count: AtomicUsize::new(1),
                weak_count: AtomicUsize::new(1),
            }))),
        }
    }

    pub fn try_unwrap(mut self: Arc<T>) -> Result<T, Arc<T>> {
        if self
            .inner()
            .strong_count
            .compare_exchange(1, 0, Release, Relaxed)
            .is_err()
        {
            return Err(self);
        }

        // When no more strong references exist, we use a fence to acquire ownership of the data, so that we can drop it
        // without any other thread accessing it.
        fence(Acquire);
        let data = self.inner_mut().data.take().unwrap();

        // After we drop the data, we check the weak count and drop the inner struct if no other weak references exist.
        if self.inner().weak_count.fetch_sub(1, Release) == 1 {
            fence(Acquire);
            unsafe { drop(Box::from_raw(self.inner.as_ptr())) };
        }

        // prevent `Arc::drop` from running (it would see `strong_count == 0` and get confused)
        std::mem::forget(self);

        Ok(data)
    }

    fn inner(&self) -> &Inner<T> {
        unsafe { self.inner.as_ref() }
    }

    fn inner_mut(&mut self) -> &mut Inner<T> {
        unsafe { self.inner.as_mut() }
    }
}

impl<T> Clone for Arc<T> {
    fn clone(&self) -> Self {
        // Use relaxed ordering because the order of this relative to other operations doesn't matter.
        self.inner().strong_count.fetch_add(1, Relaxed);

        Self { inner: self.inner }
    }
}

impl<T> Drop for Arc<T> {
    fn drop(&mut self) {
        if self.inner().strong_count.fetch_sub(1, Release) == 1 {
            // When only one reference exists, we acquire ownership of the data so that we can drop it.
            fence(Acquire);
            self.inner_mut().data.take().expect("`Arc<T>` is empty");

            if self.inner().weak_count.fetch_sub(1, Release) == 1 {
                // No other references exist, so we drop the inner struct.
                fence(Acquire);
                unsafe { drop(Box::from_raw(self.inner.as_ptr())) };
            }
        }
    }
}

impl<T> Deref for Arc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner().data.as_ref().expect("`Arc<T>` is empty")
    }
}

unsafe impl<T> Send for Arc<T> {}
unsafe impl<T> Sync for Arc<T> {}

#[derive(Debug, PartialEq)]
pub struct Weak<T> {
    inner: NonNull<Inner<T>>,
}

impl<T> Arc<T> {
    pub fn downgrade(&self) -> Weak<T> {
        self.inner().weak_count.fetch_add(1, Relaxed);
        Weak { inner: self.inner }
    }
}

impl<T> Weak<T> {
    pub fn upgrade(&self) -> Option<Arc<T>> {
        let mut strong_count = self.inner().strong_count.load(Acquire);
        if strong_count == 0 {
            None
        } else {
            loop {
                match self.inner().strong_count.compare_exchange(
                    strong_count,
                    strong_count + 1,
                    Acquire,
                    Relaxed,
                ) {
                    Ok(_) => return Some(Arc { inner: self.inner }),
                    Err(e) => strong_count = e,
                }
            }
        }
    }

    fn inner(&self) -> &Inner<T> {
        unsafe { self.inner.as_ref() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{hint, sync, thread};

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1_000))]

        #[test]
        fn test_clone(data in any::<String>()) {
            let a = Arc::new(data);
            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 1);

            let b = a.clone();
            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 2);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { b.inner.as_ref().strong_count.load(Relaxed) }, 2);
            prop_assert_eq!(unsafe { b.inner.as_ref().weak_count.load(Relaxed) }, 1);

            let c = a.clone();
            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 3);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { b.inner.as_ref().strong_count.load(Relaxed) }, 3);
            prop_assert_eq!(unsafe { b.inner.as_ref().weak_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { c.inner.as_ref().strong_count.load(Relaxed) }, 3);
            prop_assert_eq!(unsafe { c.inner.as_ref().weak_count.load(Relaxed) }, 1);

            prop_assert_eq!(&a, &b);
            prop_assert_eq!(b, c);
        }

        #[test]
        fn test_drop(data in any::<String>()) {
            let a = Arc::new(data);
            let b = a.clone();
            let c = a.clone();

            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 3);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { b.inner.as_ref().strong_count.load(Relaxed) }, 3);
            prop_assert_eq!(unsafe { b.inner.as_ref().weak_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { c.inner.as_ref().strong_count.load(Relaxed) }, 3);
            prop_assert_eq!(unsafe { c.inner.as_ref().weak_count.load(Relaxed) }, 1);

            drop(a);
            prop_assert_eq!(unsafe { b.inner.as_ref().strong_count.load(Relaxed) }, 2);
            prop_assert_eq!(unsafe { b.inner.as_ref().weak_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { c.inner.as_ref().strong_count.load(Relaxed) }, 2);
            prop_assert_eq!(unsafe { c.inner.as_ref().weak_count.load(Relaxed) }, 1);

            drop(b);
            prop_assert_eq!(unsafe { c.inner.as_ref().strong_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { c.inner.as_ref().weak_count.load(Relaxed) }, 1);
        }


        #[test]
        fn test_send(data in any::<String>()) {
            let a = Arc::new(data);
            let b = a.clone();
            let b = thread::spawn(move || b).join().unwrap();
            prop_assert_eq!(a, b);
        }

        #[test]
        fn test_deref(data in any::<String>()) {
            let a = Arc::new(data.clone());
            prop_assert_eq!(&*a, &data);
        }

        #[test]
        fn test_try_unwrap(data in any::<String>()) {
            let a = Arc::new(data.clone());
            let b = a.clone();
            let a = Arc::try_unwrap(a).unwrap_err();
            let b = Arc::try_unwrap(b).unwrap_err();
            drop(a);
            prop_assert_eq!(Arc::try_unwrap(b), Ok(data));
        }

        #[test]
        fn test_try_unwrap_multiple_threads(data in any::<String>()) {
            let a = Arc::new(data.clone());
            let b = a.clone();

            let (tx, rx) = sync::mpsc::channel();

            let h = thread::spawn(move|| {
                let mut b = Arc::try_unwrap(b).unwrap_err();

                // Signal that we've tried, and failed, to unwrap the data.
                tx.send(()).unwrap();

                loop {
                    // This will fail until `a` is dropped by the other thread.
                    match Arc::try_unwrap(b) {
                        Ok(d) => return d,
                        Err(a) => b = a,
                    }

                    hint::spin_loop();
                }
            });

            // Once the other thread has tried, and failed, to unwrap the data, `a` is dropped so that the data can be unwrapped.
            rx.recv().unwrap();
            drop(a);

            let d = h.join().unwrap();
            prop_assert_eq!(d, data);
        }

        #[test]
        fn test_downgrade(data in any::<String>()) {
            let a = Arc::new(data.clone());
            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 1);

            let w1 = a.downgrade();

            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 2);

            let w2 = a.downgrade();
            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 3);

            prop_assert_eq!(w1, w2);
        }

        #[test]
        fn test_downgrade_upgrade(data in any::<String>()) {
            let a = Arc::new(data.clone());
            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 1);

            let w = a.downgrade();
            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 1);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 2);

            let a2 = w.upgrade().unwrap();
            prop_assert_eq!(unsafe { a.inner.as_ref().strong_count.load(Relaxed) }, 2);
            prop_assert_eq!(unsafe { a.inner.as_ref().weak_count.load(Relaxed) }, 2);
            prop_assert_eq!(a, a2);
        }

        #[test]
        fn test_try_unwrap_weak(data in any::<String>()) {
            let a = Arc::new(data.clone());
            let w = a.downgrade();
            let unwrapped = a.try_unwrap().unwrap();
            prop_assert_eq!(unwrapped, data);
            prop_assert_eq!(unsafe { w.inner.as_ref().strong_count.load(Relaxed) }, 0);
            prop_assert!(w.upgrade().is_none());
        }

        #[test]
        fn test_try_unwrap_weak_multiple_threads(data in any::<String>()) {
            let a = Arc::new(data.clone());
            let b = a.clone();

            let (tx1, rx1) = sync::mpsc::channel();
            let (tx2, rx2) = sync::mpsc::channel();

            let h = thread::spawn(move|| {
                let w = b.downgrade();

                // Signal that we've downgraded the reference on this thread.
                tx1.send(()).unwrap();

                // Wait for the other thread to drop its reference.
                rx2.recv().unwrap();

                // Upgrade the weak reference.
                assert!(w.upgrade().is_some());

                // Drop this thread's reference
                drop(b);

                // Check that the weak reference is now invalid.
                assert!(w.upgrade().is_none());
            });

            // Once the other thread has downgraded the reference, drop `a`
            rx1.recv().unwrap();
            drop(a);

            // Signal that this thread has dropped its reference.
            tx2.send(()).unwrap();

            h.join().unwrap();
        }
    }
}
