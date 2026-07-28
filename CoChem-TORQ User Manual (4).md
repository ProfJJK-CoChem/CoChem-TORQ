# **CoChem-TORQ User Manual: Part 5 — Parquet Compilation, Troubleshooting & FAQs**

## **5.1 Out-of-Core Parquet Catalog Compilation (Stage 5.4)**

To prevent Pandas Out-Of-Memory crashes when handling massive spectral line lists, stream the SPCAT .cat output directly into a compressed Apache Parquet database:

\# %% \[Cell 7: Chunked Parquet Compilation\]  
compiler \= TorqCatalogCompiler(cat\_filepath="test\_spcat.cat", point\_id="001")  
success \= compiler.compile\_to\_parquet(chunk\_size=50000)  
if success:  
    print("✅ Parquet Catalog successfully compiled and locked.")

## **5.2 SpycFit Payload Export (Stage 5.5)**

Bundle all verified computational outputs, cryptographic manifests, and PGOPHER visualization skeletons into a single delivery archive for CoChem-SpycFit:

\# %% \[Cell 8: SpycFit Payload Packaging\]  
synthesizer \= TorqPayloadSynthesizer(project\_name="TargetMolecule", point\_id="001")  
synthesizer.generate\_pgopher\_skeleton()  
synthesizer.build\_manifest()  
zip\_path \= synthesizer.package\_payload()  
print(f"📦 FAIR Delivery Payload generated: {zip\_path}")

## **5.3 Troubleshooting & Common Failure Codes**

* **ZeroDivisionError in Tensor Extraction**: Triggered by a linear molecule where ![][image1]. *Resolution*: Ensure apply\_cartesian\_protections() runs prior to diagonalization; the SVD rank filter automatically catches this.  
* **OUT OF MEMORY in ORCA Execution**: Triggered during perturbative triples (T) calculations. *Resolution* : The built-in memory backoff daemon intercepts this, halves %maxcore, and successfully retries.  
* **PyArrow Compilation Warnings (\*\*\*\*\*\*\*)**: Occurs when SPCAT frequency bounds exceed Fortran display limits. *Resolution*: The compiler automatically intercepts ValueError exceptions and drops malformed lines without corrupting the wider dataset.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAaCAYAAAAT6cSuAAABqElEQVR4Xu2WvUrEQBSFN6wWoqAiIUj+04h18AXEwgcQLFZrn8PCF7AULO1tthcsLKwtLRRREGRB2EIs4rky0cl1hp00+ZH5YMjNnZPhnsxMJoOBxWJpjSiKDtEe4zguRHsW99tc2yao50aq8RWpIddoEQ9Neb4LoK4JXvhBeZ+maUz1InQkmR5h7pbnO8Ac1ZZl2bKcpBwM53JOCUSZMDfifW2TJMmRmKUKyD0YTQZEJ+JNrPK+tkFd1xpzV6r8HyB6MRK2gK62OuYKoymeQZ7n8xjnrE7je4kDzZTqU+Rnm+vyfiNQ17vKhJG5uMP7jUBtdyoTpuZKkdmZ0TCobawygdw92ifPV6AHY/3h7eBTfIxDc8PzvEXeqcDBCtir04IgWOCDyECzozFHy3XM8z+4rrskzF3wPtroyE8ohsFdxG9c0xRUYyQd2OLDVfi+vybrvil/X3iTv1y4fyoHRDxCO/0doVlo1QiDl7ieUxyG4RbXGUMDSDEdpJtyf6+RzDllTDMoSfoLjHzQFfttn8xhKXu4plzXW2BmXYRDmFypdFosFst/5wsBfpPcVbvrJAAAAABJRU5ErkJggg==>