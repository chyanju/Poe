#!/bin/bash
# run with ./run.sh <bmode> <bstart> <bend> <timeout>
# inclusive interval, 0-based

bmode=$1
bstart=$2
bend=$3
btime=$4

cnt=0

for dbenchmark in "Q0f532" "Q11ca8" "Q26ec9" "Q4bf2c" "Q5179f" "Q82b67" "Q948c4" "Qbe4fc" "Qf73e9" "Qfe843" "Q731a8" "Q98d27" "Qff1c3" "Q0ac28" "Q366c4" "Qbe9c1" "Qf0286" "Q15586" "Q2f7af" "Q48a8c" "Q5b7ad" "Q625c8" "Q74eea" "Qa256d" "Qbae4c" "Qbaf78" "Qfc9e2" "Q01296" "Q258f2" "Q292a7" "Q48d30" "Q4ad17" "Q82851" "Q88f71" "Qa3160" "Qa8df6" "Qb65c4" "Qc4f5b" "Qc8c08" "Qd52b6" "Qd54e5" "Qd6a57" "Q11ee3" "Q464cd" "Q5d606" "Q5f67e" "Q6e66f" "Q74f66" "Q7fd54" "Qa186b" "Qd35df" "Qfb9fb" "Q05b3d" "Q09b49" "Q41f90" "Q571e8" "Q5e116" "Q8a3ed" "Q92b46" "Q988ba" "Qc38bf" "Qfc407" "Q00b88" "Q0a486" "Q1567a" "Q29abb" "Q2a90c" "Q4dc67" "Q6776f" "Q840af" "Qb921e" "Qfdc88" "Q01201" "Q06745" "Q14005" "Q4b6d5" "Q6507a" "Q7180e" "Q7d51e" "Q86e6a" "Qb9a1d" "Qcb0cf" "Qd256c" "Qdb5ba" "Qe9b39" "Qeda72" "Q015ca" "Q3629d" "Q4830a" "Q4c3fb" "Q7436f" "Q77fd3" "Q8919c" "Q97935" "Qa4da8" "Qc28d5" "Qd73e2" "Qdf2b8" "Qe413c" "Qecf60" "Qf3588" "Q46e07" "Q4ec94" "Q687e9" "Q6f348" "Q78d9c" "Q7e039" "Q9559d" "Qb7071" "Qc20f7" "Qffb6a" "Q17e7e" "Q1ccf7" "Q24d16" "Q32b86" "Q3b648" "Q3e142" "Q47a13" "Q4d11d" "Q55c9b" "Q7480f" "Q88235" "Q9ea52" "Qa288c" "Qd47db" "Qf0157" "Qf0a56" "Qfef1b" "Q01e8d" "Q10efb" "Q2d890" "Q51d56" "Q9e356" "Qa85de" "Qc8e7a" "Qc9125" "Qfa0c4" "Qffc91" "Q1417b" "Q1e4db" "Q3cde2" "Q4e227" "Q746e5" "Q87e3c" "Q984df" "Qa93e3" "Qc364f" "Qc4673" "Qc797a" "Qc93f0" "Qe3d26" "Q11c9e" "Q1eebe" "Q73290" "Q78546" "Qa6baf" "Qada6c" "Qb2a38" "Qb50e8" "Qd22bb" "Qd7076" "Q02e4c" "Q0c9e0" "Q2d542" "Q2e03c" "Q414f6" "Q6604c" "Q686f1" "Q72e14" "Q73f4b" "Q915b3" "Qa0a08" "Qbb5ef" "Qf33ec" "Qf51b3" "Q0e888" "Q14e3e" "Q15046" "Q1f421" "Q2f8a0" "Q31811" "Q3b2a5" "Q3e7d6" "Q4d365" "Q528ea" "Q7026f" "Q77c58" "Qa367c" "Qa9cc2" "Qaee6b" "Qc47bc" "Qc53d5" "Qdeb10" "Qf2994" "Q194bf" "Q423f1" "Q58810" "Q59ba6" "Q65c91" "Q67064" "Q7b516" "Q859c7" "Qf7af5" "Q0d4fb" "Q12a6d" "Q14a49" "Q463a7" "Q48f58" "Q967c1" "Q994ee" "Qcc79a" "Qec647" "Qf59b3" "Q2dd9a" "Q4fb4e" "Q5b58c" "Q60b72" "Q76c08" "Q76d89" "Q79447" "Q7aad2" "Q82657" "Q88050" "Q8fc5b" "Q9ff52" "Qbfb97" "Qef0fe" "Qf877c" "Q2fea9" "Q46185" "Q4c64f" "Q71ed0" "Q721d0" "Q7f06a" "Q9c22b" "Qb2bea" "Qb7db0" "Qb9608" "Qdd9a2" "Qf4336" "Qf9fa9" "Qfd010" "Qffbc0" "Q025e1" "Q59fb0" "Qb39e8" "Qe6aed" "Qe6efa" "Q22854" "Q3f8ec" "Q3fcb9" "Q40e35" "Q43f78" "Q8358d" "Qa8811" "Qaaf34" "Qb0248" "Qb38cf" "Q01cd2" "Q51171" "Qaf628" "Qee0de" "Q0d920" "Q19f40" "Q4b352" "Q7fc9e" "Qb7427" "Qc6a7a" "Qd96be" "Qec39a" "Qf62b4" "Q00409" "Q0ca9b" "Q3b5fc" "Q5bbdb" "Q65f85" "Q9006c" "Q976e2" "Q9a2ea" "Qb5a1b" "Qc4717" "Qc7510" "Qc757c" "Qca455" "Qd1c18" "Q14105" "Q15ea6" "Q8a8c4" "Q8bc07" "Qac289" "Qf9f3f" "Q0131e" "Q0c161" "Q0e552" "Q144bc" "Q15f50" "Q18a9b" "Q18eb8" "Q1b7bf" "Q1ecd6" "Q1faaf" "Q21b73" "Q233dc" "Q3091f" "Q30f6a" "Q38219" "Q3a3d0" "Q3d093" "Q3f656" "Q42ccc" "Q434dc" "Q4476b" "Q460c7" "Q48ceb" "Q48d23" "Q4e823" "Q5c1dc" "Q66389" "Q67dde" "Q6c3f9" "Q6ece7" "Q71e80" "Q73ae0" "Q74491" "Q7b29d" "Q7f55d" "Q88e71" "Q8d669" "Q9202a" "Q98499" "Q9bbcf" "Q9d236" "Qa28dc" "Qa9457" "Qab4bb" "Qad751" "Qbf06e" "Qc19b2" "Qc65fb" "Qcd8d5" "Qd3200" "Qd68ec" "Qd9b30" "Qdeee2" "Qe5faa" "Qea225" "Qf21ed" "Qfde7b" "Q015db" "Q2c886" "Q46e3d" "Q49250" "Q49a8a" "Q62403" "Q77c0e" "Qa3e84" "Qcb7cf" "Qe15f6" "Q116c6" "Q47568" "Q9183a" "Q9aeee" "Qb1173" "Qc6cbf" "Qca5fb" "Qcfd5f" "Qd802e" "Qe0a27" "Qf6fdf" "Qfe899" "Q0ce40" "Q9d9f3" "Qd56cc" "Qdcf5a" "Q227b7" "Q28746" "Q47c6b" "Q4a2f6" "Q4ff03" "Q8bb1d" "Qaa3e3" "Qb2ca2" "Qcf419" "Qd1955" "Qf9f96" "Qfe83f" "Q0abd8" "Q0d481" "Q1edfa" "Q2347b" "Q28d6a" "Q37cd7" "Q50b63" "Q6dd3c" "Q6e89b" "Q7f966" "Q8712b" "Q8d3cf" "Qa25c9" "Qbea98" "Qd3bd0" "Qde735" "Qdea4c" "Qe578d" "Qef865" "Q01c9f" "Q0a55d" "Q895ce" "Q9d4a1" "Qb19fa" "Qc2bca" "Qc36aa" "Qc9c17" "Q01142" "Q03d1c" "Q4b197" "Q91761" "Qa37aa" "Qc7d5f" "Q2d2cd" "Q52e82" "Q547ce" "Q56379" "Q7c04a" "Qa90aa" "Qca1f9" "Qce4bc" "Qf123e" "Qf6c0d" "Q03a03" "Q132af" "Q1510c" "Q38cdb" "Q4396e" "Q499d1" "Q8add1" "Qa10f6" "Qc93fb" "Qe6754" "Qefc30" "Q00c86" "Q05963" "Q28d9d" "Q3925c" "Q39cd2" "Q4934d" "Q9a507" "Qdf507" "Qe7939" "Q383db" "Q39ea8" "Q4950a" "Q4db8e" "Q73c84" "Q742c0" "Q7645d" "Q7ae46" "Q900b1" "Q9541a" "Qa2a2a" "Qe3dc4" "Qe850d" "Qee715" "Qf1b5e" "Qfa20f" "Q4af77" "Q65bf5" "Qa51c7" "Qaf580" "Qb4668" "Qd0a26" "Qd9eef" "Qdb531" "Qdc482" "Qe2161" "Q2564e" "Q3b92b" "Q3d999" "Q6edda" "Q71bff" "Q8c373" "Q93b08" "Q9ed99" "Qbbd79" "Qf75c9" "Q0bd85" "Q1c7a1" "Q2911a" "Q6416c" "Q86b1e" "Qb6951" "Qbba20" "Qc64d3" "Qed957" "Q276e9" "Q2b404" "Q32104" "Q7e98d" "Q9ee20" "Qae0e2" "Qc3b22" "Qc6c0d" "Qcd528" "Qedbc1" "Q1217b" "Q20bda" "Q2cf88" "Q312c6" "Q38164" "Q3f735" "Q45f68" "Q5ab62" "Q60081" "Q6c7a2" "Qa977b" "Qc219b" "Qcaee6" "Qd4a8a" "Qe09b7" "Q120eb" "Q25229" "Q40081" "Q6eb85" "Qa37e7" "Qb6ed7" "Qb809d" "Qbb23f" "Q1187d" "Q1a18c" "Q27dac" "Q84eb8" "Q90986" "Qc2e8e" "Qc9547" "Qd323a" "Qee11e" "Qfdb7d" "Q00903" "Q01dff" "Q0d3ac" "Q3479e" "Q66a86" "Q68a00" "Q77d6e" "Q80675" "Q96a30" "Q9ddc0" "Qa36d4" "Qbd263" "Qbfa03" "Qeeb2d" "Qef3bb" "Q06bda" "Q14340" "Q20305" "Q21e59" "Q38735" "Q3d5be" "Q61d88" "Q948bd" "Q9e3e4" "Qa7f07" "Q527b9" "Q52b65" "Q586fa" "Q62494" "Q987cb" "Qb3627" "Qd8af7" "Qdbd75" "Qf803c" "Qfed33" "Q27c06" "Q44026" "Q55557" "Q85789" "Q916fc" "Qa986a" "Qc748b" "Qd2605" "Qe2442" "Qed628" "Q0e4fc" "Q28ab3" "Q343a9" "Q5d624" "Q86546" "Q97e44" "Qa4ec2" "Qad023" "Qb81a5" "Qc8d06" "Q0aaa2" "Q60816" "Q8c15f" "Qa2cf0" "Qc2c02" "Qc4c46" "Qc8032" "Qd780a" "Qe84a2" "Qfe6d9" "Q26b38" "Q27d85" "Q331b0" "Qc8136" "Qce7ef" "Qd9939" "Qd9a07" "Qdc525" "Qde5bd" "Qe6a46" "Q4556a" "Q6b9d5" "Q8de0c" "Q9e157" "Qb8f18" "Qbf100" "Qca502" "Qec5ee" "Q0fa44" "Q11b24" "Q528aa" "Q68bf6" "Q93b6c" "Qb0a85" "Qb184b" "Qe310a" "Qf98b7"
do
	if [[ ${cnt} -ge ${bstart} ]] && [[ ${cnt} -le ${bend} ]]
	then
		echo "running | i: ${cnt}, benchmark: ${dbenchmark}"
		python ./test_TaPas_on_VisQA_benchmark.py --benchmark ${dbenchmark} --dsl meta_visqa --skeletons visqa_normal --strategy TaPas_C --fallback auto --timeout ${btime} --mode ${bmode} > ./logs/${dbenchmark}_${bmode}.log 2>&1
	fi
	cnt=$((cnt+1))
done