# Python Ray Tracer

[From `The Ray Tracer Challenge` by Jamis Buck](http://raytracerchallenge.com/)

<p style="text-align:center;">
  <img src="./static_files//book_cover.png" alt="Cover image from the book rendered with the code in this repo" width="32%" style="margin: 5px;" />
</p>

The idea is to go through the book using `Python 3.14` as the main development language
and `Pytest` for the test suite. For the remainder of the toolchain, the plan is to
test-drive `Pyrefly` for the language server/type checker and `ruff` for linting. `uv`
is used as the package manager.

## Current Features Supported
* Primitive mathematical objects:
  * Cone
  * Cube
  * Cylinder
  * Plane
  * Sphere
  * Triangle (faceted _and_ smoothed)

* CSG Operations on objects
  * Union
  * Difference
  * Intersection
  
* Groups of objects with common transformations - allows for complex objects to be 
constructed from primitives

* Loading of Alias/Wavefront .OBJ format files

* Optimisation of triangle meshes/groups to allow for more efficient rendering (For example,
the following images exhibit a 100x improvement in render time over using the unoptimized meshes):
<p style="text-align: center;">
  <img src="./static_files/cow-no-normals.png" alt="Low resolution rendering of a cow" width="32%" style="margin: 5px;" />
  <img src="./static_files/utah-teapot.png" alt="Low resolution rendering of the Utah teapot" width="32%" style="margin: 5px;" />
  <img src="./static_files/teddy-bear.png" alt="Low resolution rendering of a teddy bear (from the rear)" width="32%" style="margin: 5px;" />
</p>

The astronaut below consists of 6381 triangle faces.  This 300x500 image rendered in 17m 41s

<p style="text-align: center;">
  <img src="./static_files/astronaut.png" alt="Smoothed mesh of an astronaut" width="32%" style="margin:5px;" />
</p>
 
* Basic materials including the following 3D patterns:
  * Stripes
  * Rings
  * Simplex Noise
  * 2 Colour Gradient
  * Checkerboard

  Patterns can be blended by way of the 'blend' pattern type which also has a bias control
  allowing the mix between the two patterns to be controlled (default is a 50/50 blend)

  Furthermore, many of the pattern types can be stacked so that rather than using simple colours
  they can include other pattern types instead

* Lights
  * Infinite point light


## To-Do:
* ~~Normal smoothing for meshes~~  DONE

* Add different light types (area lights, spotlights, directional lights); with realistic falloff.

* Allow materials to be inherited from parent Objects / Groups (currently there's no
way for loaded object meshes to have materials applied except by applying to each individual
element - which is _incredibly_ inefficient for triangle meshes!)

* Fix cone implementation - there's currently _some_ artifacts become visible within some of the
test scenes

* Simplify the APIs for material/pattern generation