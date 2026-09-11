# Public release checklist

The code organization is ready for review, but these owner decisions and
artifact checks must happen before publishing:

- [x] Select a license and add `LICENSE` (MIT).
- [ ] Verify employer/institution ownership and authority to publish.
- [ ] Confirm author list, project name, contact, and citation/BibTeX.
- [ ] Review the English backchannel phrase list for redistribution and bias.
- [ ] Publish hashes/versions for the exact Fisher and Seamless construction
      inputs without redistributing licensed data.
- [ ] Add source adapters only if their dataset licenses permit distribution.
- [ ] Run tests on a clean CPU environment and at least one real framework adapter.
- [ ] Add CI for Python 3.9--3.13, formatting, schemas, and package build.
- [ ] Compare portable rewards/advantages against frozen historical traces.
- [ ] Document known differences between historical and replacement checkpoints.
- [ ] Remove machine paths, credentials, W&B identifiers, and private artifact URIs.
- [ ] Create a security/contact policy and contribution guidelines.
- [ ] Tag the protocol and construction profile versions used for paper results.
