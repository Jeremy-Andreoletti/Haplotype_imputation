All files are for the same set of SNPs called from reads aligned the WS220 N2 reference genome
with the GATK HaplotypeCaller in joint mode. They are ordered by chromosome and physical position, 
with 'ref' being the base in the reference and 'alt' being the variant allele observed in at least 
one of the founders.

More information on variant calling and most of the metric below can be found in the
GATK documentation (https://software.broadinstitute.org/gatk/documentation/tooldocs/3.8-0/index)

Founder calls and associated quality statistics are in WS220.founder.bsqr.calls.F3.clean.gz. 
The fields (after chrom pos ref alt) are:
af.all - the alternate allele frequency across all founders
dp.all - the total read depth across all founders
qual - an aggregate quality score assigned by the SNP caller
BaseQRankSum - a rank statistic for the base qualities across all reads and samples.
    The most common format for reads is 'fastq', where each base in each reads is assigned 
    a quality score by the sequencing machine. 
MQ - a mapping quality score taking into account things like unique placement in the genome
    and the proper mapping of paired reads
FS and SOR - strand bias scores. There should be no strong bias in the DNA strand that reads align to.
QD - quality to read depth ratio
ReadPosRankSum - rank statistic based on the position of bases supporting a call within reads. 
    Base error rates tend to be much higher toward the end of reads.
MQRankSum - rank statistic based on mapping quality
ClippingRankSum - rank statistic penalising partial ('clipped') read alignments.
noCall - count of founders in which no call could be made (e.g., no aligned reads, or poor alignment)
fixed - count of homozygotes across founders
hets - count of sites that look heterozygous, and are therefore very likely to be divergence
    between the reference N2 genome and a wild-isolate at that position 
    (e.g., due to sequence duplication and divergence)


Genetic positions estimated by linear interpolation from data from a cross between N2 and a
nother founder, CB4856, are in WS220.founder.bsqr.calls.genetic.gz. 
See https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000419 for more info.

