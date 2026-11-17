# Convert the original publisher-linked RDS annotation tables without changing values.
script_arg <- grep("^--file=", commandArgs(), value=TRUE)
folder <- dirname(normalizePath(sub("^--file=", "", script_arg)))
features <- readRDS(file.path(folder, "all_CL_features.rds"))
retained <- c("Idasanutlin_6hr_expt1", "Idasanutlin_24hr_expt1",
              "Bortezomib_6hr_expt1", "Bortezomib_24hr_expt1",
              "Gemcitabine_expt10", "Prexasertib_expt10")
for (nm in retained) {
  table <- features[[nm]]
  table <- table[!is.na(table$in_pool) & table$in_pool, , drop=FALSE]
  table$source_comparison <- nm
  write.csv(table, file.path(folder, paste0(nm, "_CL_features.csv")), row.names=FALSE, na="")
}
