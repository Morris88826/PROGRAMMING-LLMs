git -C src/samples reset --hard HEAD
./run.sh -x build
./run.sh -x run_pov x.bin filein_harness
./run.sh -x build x.diff samples
./run.sh -x run_pov x.bin filein_harness
./run.sh -x run_pov exemplar_only/cpv_1/blobs/sample_solve.bin filein_harness
./run.sh -x run_pov exemplar_only/cpv_2/blobs/sample_solve.bin filein_harness
./run.sh -x run_tests