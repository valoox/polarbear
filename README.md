# Polar Bear

This is a simple Time-Series/Data Analysis library.

It mainly aims at being able to handle simple cases easily and leverage the power of NumPy
or other underlying numerical library.

Its core abstractions are:
 - The Index is a searchable 1D collection of elements, used to build queries
 - The Buffer is an n-dimensional rectangular array of values of a single type,
   typically a NumPy array (or at least behaving like one)
 - The DataSet is the conjunction of a (set of) buffer(s) with a (set of) index(ices).
   Typical dataset include the Series (one index, one buffer), the Frame (two indices,
   one buffer by datatype), etc.

This also includes all the functions required for loading and storing such items, as well
as algorithms for manipulating them in efficient ways.
