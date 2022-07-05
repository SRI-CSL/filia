# js2mlir

`js2mlir` is a standalone prototype Javascript to MLIR translator.  It
is writen in Typescript and can be built using
[NPM](https://www.npmjs.com/) and run using [Node](https://nodejs.org/).
`js2mlir` is still in early development, and only supports a small fragment
of Javascript.

To build `js2mlir`, you should install Node and NPM, then run the
following commands:

```
npm install
npm run build
```

This will compile the TypeScript and put files in the `dist` subdirectory.  You
can then run `js2mlir` on one of the test files included with Filia:

```
./bin/js2mlir script ../examples/insecure_eval.js insecure_eval.mlir
```