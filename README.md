# Python Ray Tracer

[From `The Ray Tracer Challenge` by Jamis Buck](http://raytracerchallenge.com/)

The idea is to go through the book using `Python 3.14` as the main development language
and `Pytest` for the test suite. For the remainder of the toolchain, the plan is to
test-drive `Pyrefly` for the language server/type checker and `ruff` for linting. `uv`
is used as the package manager.


## To-Do:

* Add auto-grouping / bounding volume segmentation to triangle meshes to improve rendering
performance as currently meshes take _forever_ to render (several hours for the sample obj
files in the test_toys for example)
* Research better methods of defining and rendering triangle meshes - currently just stored
as individual triangles within groups