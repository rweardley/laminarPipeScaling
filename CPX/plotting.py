import matplotlib.pyplot as plt
import numpy as np

savedir = "scaling_plots"

timestep_range = slice(100, 2000)
qps_per_rank_scale = 1
lelg = 54000  # number of elements in .re2 mesh


class Scaling_Case:
    def __init__(self, ranks_list, ranks_per_gpu, files_list, elements, polynomialorder):
        self.files_list = files_list
        self.elements = elements
        self.polynomialorder = polynomialorder
        self.time_per_timestep = []
        self.speedup = []
        self.ideal_speedup = []
        self.parallel_efficiency = []
        qps = polynomialorder**3 * elements
        self.ranks_list = ranks_list
        self.ranks_per_gpu = ranks_per_gpu
        self.qps_per_rank = [
            qps_per_rank_scale * qps / rank for rank in self.ranks_list
        ]
        self.qps_per_gpu = [
            qps_per_rank_scale * qps / (rank/ranks_per_gpu) for rank in self.ranks_list
        ]

    def scaling_calculations(self):
        print(
            "Ranks\tGPUs\tTime per timestep [s]\tSpeedup\tIdeal Speedup\tParallel Efficiency\tMean Iters (P, UVW)"
        )
        for i in range(len(self.ranks_list)):
            ranks = self.ranks_list[i]
            file = self.files_list[i]
            data = np.loadtxt(file, delimiter="\t", skiprows=1, usecols=range(7))
            time_per_timestep_column = 4
            time_per_timestep = data[timestep_range, time_per_timestep_column]
            mean_time_per_timestep = np.mean(time_per_timestep)
            if i == 0:
                reference_time = mean_time_per_timestep
                reference_ranks = ranks
            self.time_per_timestep.append(mean_time_per_timestep)
            speedup = reference_time / mean_time_per_timestep
            self.speedup.append(speedup)
            ideal_speedup = ranks / reference_ranks
            self.ideal_speedup.append(ideal_speedup)
            parallel_efficiency = (reference_time * reference_ranks) / (
                mean_time_per_timestep * ranks
            )
            self.parallel_efficiency.append(parallel_efficiency)
            print(
                f"{ranks}\t{ranks/self.ranks_per_gpu}\t{mean_time_per_timestep:.3f}"
                f"\t\t\t{speedup:.3f}\t{ideal_speedup:.3f}"
                f"\t\t{parallel_efficiency:.3f}"
                f"\t\t\t{np.mean(data[timestep_range, 5]):.3f}"
                f"\t{np.mean(data[timestep_range, 6]):.3f}"
            )

# didn't run for more than 16 ranks
ranks_N1 = [8, 12, 16]
files_N1 = [str(rank)+"_ranks/N_1.tsv" for rank in ranks_N1]
scaling_N1 = Scaling_Case(ranks_N1, 8, files_N1, lelg, 1)
print(f"\nN=1")
scaling_N1.scaling_calculations()

# didn't run for more than 16 ranks
ranks_N2 = [8, 12, 16]
files_N2 = [str(rank)+"_ranks/N_2.tsv" for rank in ranks_N2]
scaling_N2 = Scaling_Case(ranks_N2, 8, files_N2, lelg, 2)
print(f"\nN=2")
scaling_N2.scaling_calculations()

# didn't run for more than 48 ranks
ranks_N3 = [8, 12, 16, 24, 32, 40, 48]
files_N3 = [str(rank)+"_ranks/N_3.tsv" for rank in ranks_N3]
scaling_N3 = Scaling_Case(ranks_N3, 8, files_N3, lelg, 3)
print(f"\nN=3")
scaling_N3.scaling_calculations()

# up to 12 ranks failed due to 32 bit integer error
# didn't run for more than 48 ranks
ranks_N4 = [16, 24, 32, 40, 48]
files_N4 = [str(rank)+"_ranks/N_4.tsv" for rank in ranks_N4]
scaling_N4 = Scaling_Case(ranks_N4, 8, files_N4, lelg, 4)
print(f"\nN=4")
scaling_N4.scaling_calculations()

# up to 16 ranks failed due to 32 bit integer error
ranks_N5 = [24, 32, 40, 48, 56, 64]
files_N5 = [str(rank)+"_ranks/N_5.tsv" for rank in ranks_N5]
scaling_N5 = Scaling_Case(ranks_N5, 8, files_N5, lelg, 5)
print(f"\nN=5")
scaling_N5.scaling_calculations()

# up to 24 ranks failed due to 32 bit integer error
ranks_N6 = [32, 40, 48, 56, 64]
files_N6 = [str(rank)+"_ranks/N_6.tsv" for rank in ranks_N6]
scaling_N6 = Scaling_Case(ranks_N6, 8, files_N6, lelg, 6)
print(f"\nN=6")
scaling_N6.scaling_calculations()

# up to 40 ranks failed due to 32 bit integer error
ranks_N7 = [48, 56, 64]
files_N7 = [str(rank)+"_ranks/N_7.tsv" for rank in ranks_N7]
scaling_N7 = Scaling_Case(ranks_N7, 8, files_N7, lelg, 7)
print(f"\nN=7")
scaling_N7.scaling_calculations()

# up to 56 ranks failed due to 32 bit integer error
ranks_N8 = [64]
files_N8 = [str(rank)+"_ranks/N_8.tsv" for rank in ranks_N8]
scaling_N8 = Scaling_Case(ranks_N8, 8, files_N8, lelg, 8)
print(f"\nN=8")
scaling_N8.scaling_calculations()

plt.plot(ranks_N3, scaling_N3.time_per_timestep, "x-", label=r"$N=3$")
plt.plot(ranks_N4, scaling_N4.time_per_timestep, "x-", label=r"$N=4$")
plt.plot(ranks_N5, scaling_N5.time_per_timestep, "x-", label=r"$N=5$")
plt.plot(ranks_N6, scaling_N6.time_per_timestep, "x-", label=r"$N=6$")
plt.plot(ranks_N7, scaling_N7.time_per_timestep, "x-", label=r"$N=7$")
plt.plot(ranks_N8, scaling_N8.time_per_timestep, "x-", label=r"$N=8$")
plt.xlabel("Ranks")
plt.ylabel("Walltime per timestep [s]")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/time_per_timestep_ranks.png", bbox_inches="tight")
plt.close()

plt.plot(ranks_N3, scaling_N3.speedup, "x-", label=r"$N=3$")
plt.plot(ranks_N4, scaling_N4.speedup, "x-", label=r"$N=4$")
plt.plot(ranks_N5, scaling_N5.speedup, "x-", label=r"$N=5$")
plt.plot(ranks_N6, scaling_N6.speedup, "x-", label=r"$N=6$")
plt.plot(ranks_N7, scaling_N7.speedup, "x-", label=r"$N=7$")
plt.plot(ranks_N8, scaling_N8.speedup, "x-", label=r"$N=8$")
plt.plot(ranks_N3, scaling_N3.ideal_speedup, "k--", label=r"Ideal")
plt.plot(ranks_N4, scaling_N4.ideal_speedup, "k--")
plt.plot(ranks_N5, scaling_N5.ideal_speedup, "k--")
plt.plot(ranks_N6, scaling_N6.ideal_speedup, "k--")
plt.plot(ranks_N7, scaling_N7.ideal_speedup, "k--")
plt.plot(ranks_N8, scaling_N8.ideal_speedup, "k--")
plt.xlabel("Ranks")
plt.ylabel("Speedup")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/speedup_ranks.png", bbox_inches="tight")
plt.close()

plt.plot(ranks_N3, scaling_N3.parallel_efficiency, "x-", label=r"$N=3$")
plt.plot(ranks_N4, scaling_N4.parallel_efficiency, "x-", label=r"$N=4$")
plt.plot(ranks_N5, scaling_N5.parallel_efficiency, "x-", label=r"$N=5$")
plt.plot(ranks_N6, scaling_N6.parallel_efficiency, "x-", label=r"$N=6$")
plt.plot(ranks_N7, scaling_N7.parallel_efficiency, "x-", label=r"$N=7$")
plt.plot(ranks_N8, scaling_N8.parallel_efficiency, "x-", label=r"$N=8$")
plt.xlabel("Ranks")
plt.ylabel("Parallel Efficiency")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/efficiency_ranks.png", bbox_inches="tight")
plt.close()

plt.plot(scaling_N3.qps_per_rank, scaling_N3.time_per_timestep, "x-", label=r"$N=3$")
plt.plot(scaling_N4.qps_per_rank, scaling_N4.time_per_timestep, "x-", label=r"$N=4$")
plt.plot(scaling_N5.qps_per_rank, scaling_N5.time_per_timestep, "x-", label=r"$N=5$")
plt.plot(scaling_N6.qps_per_rank, scaling_N6.time_per_timestep, "x-", label=r"$N=6$")
plt.plot(scaling_N7.qps_per_rank, scaling_N7.time_per_timestep, "x-", label=r"$N=7$")
plt.plot(scaling_N8.qps_per_rank, scaling_N8.time_per_timestep, "x-", label=r"$N=8$")
plt.xlabel(r"Quadrature Points per Rank ($n=EN^3$)")
plt.gca().invert_xaxis()
plt.ylabel("Walltime per timestep [s]")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/time_per_timestep_qps_ranks.png", bbox_inches="tight")
plt.close()

plt.plot(scaling_N3.qps_per_gpu, scaling_N3.time_per_timestep, "x-", label=r"$N=3$")
plt.plot(scaling_N4.qps_per_gpu, scaling_N4.time_per_timestep, "x-", label=r"$N=4$")
plt.plot(scaling_N5.qps_per_gpu, scaling_N5.time_per_timestep, "x-", label=r"$N=5$")
plt.plot(scaling_N6.qps_per_gpu, scaling_N6.time_per_timestep, "x-", label=r"$N=6$")
plt.plot(scaling_N7.qps_per_gpu, scaling_N7.time_per_timestep, "x-", label=r"$N=7$")
plt.plot(scaling_N8.qps_per_gpu, scaling_N8.time_per_timestep, "x-", label=r"$N=8$")
plt.xlabel(r"Quadrature Points per GPU ($n=EN^3$)")
plt.gca().invert_xaxis()
plt.ylabel("Walltime per timestep [s]")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/time_per_timestep_qps_gpu.png", bbox_inches="tight")
plt.close()

plt.plot(scaling_N3.qps_per_rank, scaling_N3.speedup, "x-", label=r"$N=3$")
plt.plot(scaling_N4.qps_per_rank, scaling_N4.speedup, "x-", label=r"$N=4$")
plt.plot(scaling_N5.qps_per_rank, scaling_N5.speedup, "x-", label=r"$N=5$")
plt.plot(scaling_N6.qps_per_rank, scaling_N6.speedup, "x-", label=r"$N=6$")
plt.plot(scaling_N7.qps_per_rank, scaling_N7.speedup, "x-", label=r"$N=7$")
plt.plot(scaling_N8.qps_per_rank, scaling_N8.speedup, "x-", label=r"$N=8$")
plt.plot(scaling_N3.qps_per_rank, scaling_N3.ideal_speedup, "k--", label=r"Ideal")
plt.plot(scaling_N4.qps_per_rank, scaling_N4.ideal_speedup, "k--")
plt.plot(scaling_N5.qps_per_rank, scaling_N5.ideal_speedup, "k--")
plt.plot(scaling_N6.qps_per_rank, scaling_N6.ideal_speedup, "k--")
plt.plot(scaling_N7.qps_per_rank, scaling_N7.ideal_speedup, "k--")
plt.plot(scaling_N8.qps_per_rank, scaling_N8.ideal_speedup, "k--")
plt.xlabel(r"Quadrature Points per Rank ($n=EN^3$)")
plt.gca().invert_xaxis()
plt.ylabel("Speedup")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/speedup_qps_rank.png", bbox_inches="tight")
plt.close()

plt.plot(scaling_N3.qps_per_gpu, scaling_N3.speedup, "x-", label=r"$N=3$")
plt.plot(scaling_N4.qps_per_gpu, scaling_N4.speedup, "x-", label=r"$N=4$")
plt.plot(scaling_N5.qps_per_gpu, scaling_N5.speedup, "x-", label=r"$N=5$")
plt.plot(scaling_N6.qps_per_gpu, scaling_N6.speedup, "x-", label=r"$N=6$")
plt.plot(scaling_N7.qps_per_gpu, scaling_N7.speedup, "x-", label=r"$N=7$")
plt.plot(scaling_N8.qps_per_gpu, scaling_N8.speedup, "x-", label=r"$N=8$")
plt.plot(scaling_N3.qps_per_gpu, scaling_N3.ideal_speedup, "k--", label=r"Ideal")
plt.plot(scaling_N4.qps_per_gpu, scaling_N4.ideal_speedup, "k--")
plt.plot(scaling_N5.qps_per_gpu, scaling_N5.ideal_speedup, "k--")
plt.plot(scaling_N6.qps_per_gpu, scaling_N6.ideal_speedup, "k--")
plt.plot(scaling_N7.qps_per_gpu, scaling_N7.ideal_speedup, "k--")
plt.plot(scaling_N8.qps_per_gpu, scaling_N8.ideal_speedup, "k--")
plt.xlabel(r"Quadrature Points per GPU ($n=EN^3$)")
plt.gca().invert_xaxis()
plt.ylabel("Speedup")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/speedup_qps_gpu.png", bbox_inches="tight")
plt.close()

plt.plot(scaling_N3.qps_per_rank, scaling_N3.parallel_efficiency, "x-", label=r"$N=3$")
plt.plot(scaling_N4.qps_per_rank, scaling_N4.parallel_efficiency, "x-", label=r"$N=4$")
plt.plot(scaling_N5.qps_per_rank, scaling_N5.parallel_efficiency, "x-", label=r"$N=5$")
plt.plot(scaling_N6.qps_per_rank, scaling_N6.parallel_efficiency, "x-", label=r"$N=6$")
plt.plot(scaling_N7.qps_per_rank, scaling_N7.parallel_efficiency, "x-", label=r"$N=7$")
plt.plot(scaling_N8.qps_per_rank, scaling_N8.parallel_efficiency, "x-", label=r"$N=8$")
plt.xlabel(r"Quadrature Points per Rank ($n=EN^3$)")
plt.gca().invert_xaxis()
plt.ylabel("Parallel Efficiency")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/efficiency_qps_rank.png", bbox_inches="tight")
plt.close()

plt.plot(scaling_N3.qps_per_gpu, scaling_N3.parallel_efficiency, "x-", label=r"$N=3$")
plt.plot(scaling_N4.qps_per_gpu, scaling_N4.parallel_efficiency, "x-", label=r"$N=4$")
plt.plot(scaling_N5.qps_per_gpu, scaling_N5.parallel_efficiency, "x-", label=r"$N=5$")
plt.plot(scaling_N6.qps_per_gpu, scaling_N6.parallel_efficiency, "x-", label=r"$N=6$")
plt.plot(scaling_N7.qps_per_gpu, scaling_N7.parallel_efficiency, "x-", label=r"$N=7$")
plt.plot(scaling_N8.qps_per_gpu, scaling_N8.parallel_efficiency, "x-", label=r"$N=8$")
plt.xlabel(r"Quadrature Points per GPU ($n=EN^3$)")
plt.gca().invert_xaxis()
plt.ylabel("Parallel Efficiency")
plt.legend(frameon=False)
plt.savefig(f"{savedir}/efficiency_qps_gpu.png", bbox_inches="tight")
plt.close()