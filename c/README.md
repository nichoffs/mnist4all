This is a rewrite -- I had to clarify and clean some things up.

A `Buffer` is the data structure underlying `Tensor`, similar to how `np.ndarray` underlied `Tensor` in the python version.
It has two properties:

- `float *data` - a contiguous piece of memory
- `ShapeTracker st` - represents a "view" of the data - defines how operations occur on data and with which other buffers the buffer can interact

A new `ShapeTracker` is created for every new `Buffer`, although data may remain constant when possible, i.e. movement ops.
