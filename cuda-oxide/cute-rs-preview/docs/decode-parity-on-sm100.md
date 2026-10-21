
In a nutshell

- **Smarter synchronization and bookkeeping:** Added compiler support and typed pipelines that avoid unnecessary cache invalidations. Reused memory descriptors and reduced repeated calculations.
- **Wider output stores:** Added typed, aligned 256-bit copies and their compiler lowering—twice the data per store. This closed the remaining gap while keeping alignment safety in Rust’s types.