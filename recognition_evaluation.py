from src.workflow import workflow

#workflow([(1, 8), (2, 8), (3, 8), (4, 8)], methode="LBP_uniform")

workflow([(1, 8), (2, 8), (2, 12), (3, 8), (3, 16), (4, 8), (8, 16)], methode="LBP_histogram_8x8")

"""
workflow([(1, 8), (2, 8), (2, 12), (3, 8), (3, 16), (4, 8), (8, 16)], methode="LBP_histogram_4x4")

workflow([(1, 8), (2, 8), (2, 12), (3, 8), (3, 16), (4, 8), (8, 16)], methode="LBP_histogram_16x16")
workflow([(1, 8), (2, 8), (2, 12), (3, 8), (3, 16), (4, 8), (8, 16)], methode="sklearn_lbp")

workflow([(1, 8), (2, 8), (3, 8), (4, 8)], methode="LBP_histogram_16x16", width=64, height=64)

workflow([(2, 8)], methode="LBP", width=64, height=64)

workflow([(2, 8), (3, 8), (4, 8)], methode="LBP_histogram_8x8", width=64, height=64)

workflow([(1, 8), (2, 8), (3, 8), (4, 8)], methode="LBP_histogram_16x16", width=64, height=128)

workflow([(2, 8)], methode="LBP", width=64, height=128)

workflow([(2, 8), (3, 8), (4, 8)], methode="LBP_histogram_8x8", width=64, height=128)

workflow([(0, 0)], methode="pixel_by_pixel", width=128, height=128)"""