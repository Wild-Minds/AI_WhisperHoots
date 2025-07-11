### What can an AI-trained language software tell us about chimpanzee calls?
### Script and data available at : https://osf.io/ymazb/?view_only=36bf1fa44702409783644cf824ddf134
### Last update 13.01.2025

# R packages used
library(tidyverse); library(tidytext); library(readxl); library(dplyr); library(ggplot2); library(openxlsx);library(car); library(effects); library(cowplot)
library(lme4); library(performance); library(effects); library(ggeffects); library(jtools); library(ggpubr); library(broom)

# Plot settings
My_label_2 <- theme(plot.title = element_text(size = 16, hjust = 0.5, family = "Times", face = "bold"),
                    axis.title.x = element_text(size = 16, family = "Times"),
                    axis.text.x = element_text(size = 16, family = "Times"),
                    axis.text.y = element_text(size = 16, family = "Times"),
                    axis.title.y = element_text(size = 16, family = "Times"))

### Sentiment analysis: ----
# Read the Excel file containing transcribed words from Whisper
data <- read_excel("Data for AI analyses.xlsx", sheet = "Transcriptions")

# Obtain tokenised words
# Replace parentheses, square brackets, specified punctuation, commas, quotation marks, specific hyphen, and the specific character with spaces. Split phrases into individual words
data_tokens <- data %>%
  mutate(Final_translation = tolower(Final_translation),  # Convert text to lowercase
         Final_translation = str_replace_all(Final_translation, "[\\[\\]()!,.?…,\"'‘’“”–\\-\\u200B]", " "),  # Replace specified characters with spaces
         Final_translation = str_replace_all(Final_translation, "\\b\\*|\\*\\b", "" ), # Remove asterisks attached to words
         tokenized_words = str_split(Final_translation, pattern = "\\s+")) %>%
  unnest(tokenized_words) %>%
  filter(tokenized_words != "") %>%  # Remove rows with empty cells
  select(Identifier, Final_translation, tokenized_words)

# Check tokenised words
View(data_tokens)

#### BING lexicon (n = 6786 words) ----
# Load the sentiment lexicon
data("sentiments")

# Export lexicon to inspect
bing_lexicon <- get_sentiments("bing")
# write.xlsx(bing_lexicon, "BING lexicon.xlsx")

# Perform sentiment analysis for each Identifier and tokenized word
sentiment_scores_bing1 <- data_tokens %>%
  inner_join(get_sentiments("bing"), by = c("tokenized_words" = "word")) %>%
  group_by(Identifier, tokenized_words) %>%
  summarise(avg_sentiment_score = mean(sentiment == "positive") - mean(sentiment == "negative"))

# View scores
View(sentiment_scores_bing1)

# Calculate the number of tokenized words scored
num_tokenized_words <- data_tokens %>% 
  summarise(total_tokenized_words = n())

# Visualise sentiment score per each word. Plot sentiment scores
ggplot(sentiment_scores_bing1, aes(x = avg_sentiment_score)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Sentiment Scores",
       x = "Sentiment Score",
       y = "Frequency") +
  theme_minimal()

# Write tokenised words with bing sentiment score in a csv file
# write.csv(sentiment_scores_bing1, file = "sentiment scores BING.csv")


#### AFINN lexicon (n = 2477 words) ----
library("textdata")
afinn_lexicon <- get_sentiments("afinn")

# Export lexicon to inspect
write.xlsx(afinn_lexicon, "AFINN lexicon.xlsx")

# Merge tokenized data with AFINN lexicon and include tokenized words
data_sentiment_afinn <- data_tokens %>%
  left_join(afinn_lexicon, by = c("tokenized_words" = "word")) %>%
  group_by(Identifier, tokenized_words) %>%
  summarise(sentiment_afinn = sum(value, na.rm = TRUE))

# View the scores
View(data_sentiment_afinn)

# Write tokenised words with afinn sentiment score in a csv file
# write.csv(average_scores_AFINN, file = "sentiment scores AFINN.csv")

#### Combine the two csv files from BING and AFINN, remove duplicates, standardise score and calculate average: done on Excel file "Data for AI analyses", sheet = Sentiment scores ----

### Test distribution of positive and negative scores in lexicons ----
# Load datasets
afinn <- read_excel("AFINN lexicon_1.xlsx")
bing <- read_excel("BING lexicon_1.xlsx")

# Count scores for each dataset
afinn_counts <- table(afinn$score)
bing_counts <- table(bing$score)

# Define expected proportions
expected_proportions <- c(0.5, 0.5) # Assuming equal distribution of -1 and +1

# Total counts for each dataset
afinn_total <- sum(afinn_counts)
bing_total <- sum(bing_counts)

# Expected counts
afinn_expected <- expected_proportions * afinn_total
bing_expected <- expected_proportions * bing_total

# Chi-square goodness-of-fit test for AFINN
afinn_chisq <- chisq.test(x = afinn_counts, p = expected_proportions)
cat("AFINN Chi-Square Test:\n")
print(afinn_chisq)

# Chi-square goodness-of-fit test for BING
bing_chisq <- chisq.test(x = bing_counts, p = expected_proportions)
cat("\nBING Chi-Square Test:\n")
print(bing_chisq)


### Plot word cloud ----
# R package
library(wordcloud2)

# Count the frequency of each tokenized word (N = 522 tokenised words)
word_freq <- data_tokens %>%
  count(tokenized_words) %>%
  rename(Word = tokenized_words, Freq = n)

# Create word cloud plot with adjusted dimensions
wordcloud_plot <- wordcloud2(word_freq, size = 3, minRotation = -pi/6, maxRotation = pi/6, rotateRatio = 0.5, gridSize= 3)
# Save as html link, then export as PDF
# Create file with list of all tokenised words
write.csv(word_freq, file = "list of tokenised word frequency.csv")


### Contextual variation of sentiment scores ----
dataCS <- read_excel("Data for AI analyses.xlsx", sheet = "Sentiment scores")
chisq_result <- chisq.test(dataCS$Converted_score, dataCS$Context)
print(chisq_result)


### Models to investigate relationships between recording features and ability to transcribe words ----
# Call data
dataMod <- read_excel("Data for AI analyses.xlsx", sheet = "Transcriptions with details")

dataMod$Transcribed_words= as.factor(dataMod$Transcribed_words)
dataMod$Database_size= as.numeric(dataMod$Database_size)
dataMod$Context= as.factor(dataMod$Context)
dataMod$Context=relevel(dataMod$Context, ref="R") 

#### Model 1: success in transcribing words depending on duration, nr. of units and context ----
Mod1 <- glm(Transcribed_words ~ Duration_whole_ + Number_units + Context, data=dataMod, family=binomial)
S(Mod1)
plot(allEffects(Mod1))

check_collinearity(Mod1) # Low correlation (max 1.64)

# Extract the summary statistics:
model_summary_Mod1 <- tidy(Mod1)

# Extract the confidence intervals
CI_Mod1 <- confint(Mod1) # OK
CI_Mod1 <- as.data.frame(CI_Mod1)
names(CI_Mod1) <- c("CI low", "CI high")

# Combine the summary statistics and confidence intervals
model_summary_Mod1_c <- cbind(model_summary_Mod1, CI_Mod1)

# Export to CSV
# write.csv(model_summary_Mod1_c, "Mod1 summary.csv", row.names = FALSE)

# Plot significant effect:
My_label_2 <- theme(plot.title = element_text(size = 14, hjust = 0.5, family = "Times", face = "bold"),
                    axis.title.x = element_text(size = 14, family = "Times"),
                    axis.text.x = element_text(size = 14, family = "Times"),
                    axis.text.y = element_text(size = 14, family = "Times"),
                    axis.title.y = element_text(size = 14, family = "Times"))

plot_Mod1 <- effect_plot(Mod1, pred = "Number_units", data = dataMod, plot.points = T, interval = TRUE, int.type = "confidence",
                         x.label = "Number of pant hoot units", y.label = "", line.thickness = 1.7, point.size = 0.7, point.alpha = 0.4, jitter = 0.1, colors = "black", line.colors = "purple1") + 
  scale_y_continuous(breaks = seq(0, 1, 0.10), name = "Probability of transcribing words") +
  theme_bw() +
  theme(panel.border = element_blank(), axis.line = element_line(colour = "black"))+ 
  theme(legend.position = "none")+
  My_label_2
plot_Mod1

#### Model 2: number of words transcribed depending on duration, nr of units and context ----
Mod2 <- lm(Number_words ~Duration_whole_ + Number_units + Context, data=dataMod)
S(Mod2)
Anova(Mod2, type= "II") # no significant results

check_collinearity(Mod2) # Low correlation (max 1.43)

# Extract the summary statistics:
model_summary_Mod2 <- tidy(Mod2)

# Extract the confidence intervals
CI_Mod2 <- confint(Mod2) # OK
CI_Mod2 <- as.data.frame(CI_Mod2)
names(CI_Mod2) <- c("CI low", "CI high")

# Combine the summary statistics and confidence intervals
model_summary_Mod2_c <- cbind(model_summary_Mod2, CI_Mod2)

# Export to CSV
# write.csv(model_summary_Mod2_c, "Mod2 summary.csv", row.names = FALSE)

#### Model 3: succes in transcribing words depending on phase properties ----
Mod3 <- glm(Transcribed_words ~ Duration_intro + Duration_BU + Duration_Climax + Duration_LD + Number_Intro_units + Number_BU_units + Number_Climax_units + Number_LD_units + Context, data=dataMod, family=binomial)
S(Mod3)
plot(allEffects(Mod3))
check_collinearity(Mod3) # Low to Moderate correlation (<10 so OK)

# Extract the summary statistics:
model_summary_Mod3 <- tidy(Mod3)

# Extract the confidence intervals
CI_Mod3 <- confint(Mod3) # OK
CI_Mod3 <- as.data.frame(CI_Mod3)
names(CI_Mod3) <- c("CI low", "CI high")

# Combine the summary statistics and confidence intervals
model_summary_Mod3_c <- cbind(model_summary_Mod3, CI_Mod3)

# Export to CSV
# write.csv(model_summary_Mod3_c, "Mod3 summary.csv", row.names = FALSE)

# Plot significant effects:
# Plot 1:
plot_Mod3a <- effect_plot(Mod3, pred = "Number_Intro_units", data = dataMod, plot.points = T, interval = TRUE, int.type = "confidence",
                         x.label = "Number of Introduction units", y.label = "", line.thickness = 1.7, point.size = 0.7, point.alpha = 0.4, jitter = 0.1, colors = "black", line.colors = "red2") + 
  scale_y_continuous(breaks = seq(0, 1, 0.10), name = "") +
  theme_bw() +
  theme(panel.border = element_blank(), axis.line = element_line(colour = "black"))+ 
  theme(legend.position = "none")+
  My_label_2
plot_Mod3a

# Plot 2:
plot_Mod3b <- effect_plot(Mod3, pred = "Number_BU_units", data = dataMod, plot.points = T, interval = TRUE, int.type = "confidence",
                          x.label = "Number of Build-up units", y.label = "", line.thickness = 1.7, point.size = 0.7, point.alpha = 0.4, jitter = 0.1, colors = "black", line.colors = "green3") + 
  scale_y_continuous(breaks = seq(0, 1, 0.10), name = "") +
  theme_bw() +
  theme(panel.border = element_blank(), axis.line = element_line(colour = "black"))+ 
  theme(legend.position = "none")+
  My_label_2
plot_Mod3b

# Plot 3:
plot_Mod3c <- effect_plot(Mod3, pred = "Number_Climax_units", data = dataMod, plot.points = T, interval = TRUE, int.type = "confidence",
                          x.label = "Number of Climax units", y.label = "", line.thickness = 1.7, point.size = 0.7, point.alpha = 0.4, jitter = 0.1, colors = "black", line.colors = "blue2") + 
  scale_y_continuous(breaks = seq(0, 1, 0.10), name = "Probability of transcribing words") +
  theme_bw() +
  theme(panel.border = element_blank(), axis.line = element_line(colour = "black"))+ 
  theme(legend.position = "none")+
  My_label_2
plot_Mod3c

# Plot 4:
plot_Mod3d <- effect_plot(Mod3, pred = "Context", data = dataMod, plot.points = T, interval = TRUE, int.type = "confidence",
                          x.label = "Context", y.label = "", line.thickness = 1.7, point.size = 0.7, point.alpha = 0.4, jitter = 0.1, colors = "black", line.colors = "orange2") + 
  scale_y_continuous(breaks = seq(0, 1, 0.10), name = "") +
  theme_bw() +
  theme(panel.border = element_blank(), axis.line = element_line(colour = "black"))+ 
  theme(legend.position = "none")+
  scale_x_discrete(labels = c("Rest", "Feed", "Travel")) +
  My_label_2
plot_Mod3d

# Combine two plots
ggarrange(plot_Mod1, plot_Mod3a, plot_Mod3b, plot_Mod3c, plot_Mod3d, ncol = 3, nrow = 2, labels = c("A)", "B)", "C)", "D)", "E)"), font.label = list(size = 16, color = "red", face = "bold", family = "Times"), label.y = c(1, 1, 1, 1,1), label.x = c(0.08, 0.08, 0.08, 0.08,0.08)) 
# Export 800x710

#### Model 4: number of words transcribed depending on phase properties ---- 
Mod4 <- lm(Number_words ~ Duration_intro + Duration_BU + Duration_Climax + Duration_LD + Number_Intro_units + Number_BU_units + Number_Climax_units + Number_LD_units + Context, data=dataMod)
S(Mod4)
Anova(Mod4, type= "II") # No significant results

check_collinearity(Mod4) # Low and Moderate correlation

# Extract the summary statistics
model_summary_Mod4 <- tidy(Mod4)

# Extract the confidence intervals
CI_Mod4 <- confint(Mod4) # OK
CI_Mod4 <- as.data.frame(CI_Mod4)
names(CI_Mod4) <- c("CI low", "CI high")

# Combine the summary statistics and confidence intervals
model_summary_Mod4_c <- cbind(model_summary_Mod4, CI_Mod4)

# Export to CSV
# write.csv(model_summary_Mod4_c, "Mod4 summary.csv", row.names = FALSE)

#### Model 5: probability of transcription depending on database size ---- 
Mod5 <- glm(Transcribed_words ~ Database_size, data=dataMod, family=binomial)
S(Mod5)
plot(allEffects(Mod5)) # huge confidence bands as the database grows.
# Check confidence intervals:
confint(Mod5) 

# Extract the summary statistics:
model_summary_Mod5 <- tidy(Mod5)

# Extract the confidence intervals
CI_Mod5 <- confint(Mod5) # OK
CI_Mod5 <- as.data.frame(CI_Mod5)
names(CI_Mod5) <- c("CI low", "CI high")

# Combine the summary statistics and confidence intervals
model_summary_Mod5_c <- cbind(model_summary_Mod5, CI_Mod5)

# Export to CSV
# write.csv(model_summary_Mod5_c, "Mod5 summary.csv", row.names = FALSE)

#### Model 6: number of words transcribed depending on database size ---- 
Mod6 <- lm(Number_words ~ Database_size, data=dataMod)
S(Mod6)

# Extract the summary statistics
model_summary_Mod6 <- tidy(Mod6)

# Extract the confidence intervals
CI_Mod6 <- confint(Mod6) # OK
CI_Mod6 <- as.data.frame(CI_Mod6)
names(CI_Mod6) <- c("CI low", "CI high")

# Combine the summary statistics and confidence intervals
model_summary_Mod6_c <- cbind(model_summary_Mod6, CI_Mod6)

# Export to CSV
# write.csv(model_summary_Mod6_c, "Mod6 summary.csv", row.names = FALSE)

# Combine two plots
ggarrange(Zipf1Plot, Zipf2plot, ncol = 2, nrow = 1, labels = c("A)", "B)"), font.label = list(size = 16, color = "red", face = "bold", family = "Times"), label.y = c(1, 1), label.x = c(0.05, 0.05)) 
# Export 750x550¨


### Visualise language differences ----
# Load the data
dataLang <- read_excel("Data for AI analyses.xlsx", sheet = "Transcriptions with details")

# Calculate the proportion of Yes and No for each language
prop_transcribed <- by(dataLang$Transcribed_words, dataLang$Language, FUN = function(x) prop.table(table(x)))

# Calculate the proportion of Yes and No for each language
prop_transcribed <- dataLang %>%
  group_by(Language) %>%
  summarize(Yes = sum(Transcribed_words == "Yes", na.rm = TRUE) / sum(!is.na(Transcribed_words)),
            No = sum(Transcribed_words == "No", na.rm = TRUE) / sum(!is.na(Transcribed_words)))

# Convert data to long format for plotting
prop_transcribed_long <- prop_transcribed %>%
  tidyr::pivot_longer(cols = c("Yes", "No"), names_to = "Transcribed_words", values_to = "Proportion")

# info for plotting:
My_label_2 <- theme(plot.title = element_text(size = 16, hjust = 0.5, family = "Times", face = "bold"),
                    axis.title.x = element_text(size = 16, family = "Times"),
                    axis.text.x = element_text(size = 16, family = "Times"),
                    axis.text.y = element_text(size = 16, family = "Times"),
                    axis.title.y = element_text(size = 16, family = "Times"))

# Create the plot of proportion of transcribed words per language
PropLang <- ggplot(prop_transcribed_long, aes(x = Language, y = Proportion, fill = Transcribed_words)) +
  geom_bar(stat = "identity") +
  labs(y = "Proportion",
       fill = "Transcribed") +
  theme_minimal() +
  My_label_2 +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "none")
PropLang

# Create the plot for number of words transcribed per language
NWordLang <- ggplot(dataLang, aes(x = Language, y = Number_words)) +
  geom_bar(stat = "identity", fill = "#1f78b4", color = "transparent") +
  labs(y = "Number of words") +
  theme_minimal() + 
  My_label_2 +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) 
NWordLang

# Combine two plots
ggarrange(PropLang, NWordLang, ncol = 2, nrow = 1, labels = c("A)", "B)"), font.label = list(size = 16, color = "red", face = "bold", family = "Times"), label.y = c(1, 1), label.x = c(0.18, 0.18)) 
# Export 750x550 

### Visualise database size (hours) per language: ----
# Create a data frame with the provided data
data <- data.frame(
  Language = c("Spanish", "English", "Chinese", "Swahili", "Arabic", "Indonesian", "Tamil", "Turkish"),
  Database_size = c(11100, 438218, 23446, 5.4, 739, 1014, 136, 4333)
)

# Normal distribution:
# Create the bar plot
plot_database <- ggplot(data, aes(x = Language, y = Database_size)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  labs(x = "Language",
       y = "Database size (hours)")+
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  My_label_2
plot_database

# Apply logarithmic scale to help visualise the differences between databases: 
# Bar plot with logarithmic scale
plot_database_log <- ggplot(data, aes(x = Language, y = Database_size)) +
  geom_bar(stat = "identity", fill = "orange") +
  scale_y_log10() +
  theme_minimal() +
  labs(x = "Languages",
       y = "Whisper database size (hours) (log)") +
  theme_minimal() + 
  My_label_2
plot_database_log

# Create the bar plot with logarithmic scale and value labels at the base
plot_database_log <- ggplot(data, aes(x = Language, y = Database_size)) +
  geom_bar(stat = "identity", fill = "orange") +
  geom_text(aes(label = Database_size), vjust = 1.5, size = 3) +  
  scale_y_log10() +  # Apply logarithmic scale
  theme_minimal() +
  labs(x = "Language",
       y = "Database size (hours) (log)") +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(angle = 45, hjust = 1))+ 
  My_label_2
plot_database_log

# Combine two plots
ggarrange(plot_database, plot_database_log, ncol = 2, nrow = 1, labels = c("A)", "B)"), font.label = list(size = 16, color = "red", face = "bold", family = "Times"), label.y = c(1, 1), label.x = c(0.18, 0.18)) 
# Export 750x550 

### Non-word transcriptions: ----
data_sounds <- read_excel("Data for AI analyses.xlsx", sheet = "Transcriptions with details")

# Packages: 
library(dplyr); library(stringr); library(tidyr)

# Step 1: Extract text within brackets and tokenize
data_tokens_sounds <- data_sounds %>%
  mutate(Final_transcription = tolower(Final_translation),  # Convert text to lowercase
         words_in_brackets = str_extract_all(Final_translation, "\\[([^]]+)\\]|\\(([^)]+)\\)")) %>%  # Extract text within [] or ()
  unnest(words_in_brackets) %>%
  mutate(words_in_brackets = str_replace_all(words_in_brackets, "[\\[\\]()]", ""),  # Remove brackets
         tokenized_words = str_split(words_in_brackets, "\\s+")) %>%  # Split phrases into individual words
  unnest(tokenized_words) %>%
  filter(tokenized_words != "") %>%  # Remove rows with empty cells
  select(Identifier, tokenized_words)  # Select the relevant columns

# Print the resulting dataframe
print(data_tokens_sounds)

#### Word plot of sounds:----
# Count the frequency of each tokenized sound
sound_freq <- data_tokens_sounds %>%
  count(tokenized_words) %>%
  rename(Word = tokenized_words, Freq = n)

# Create word cloud plot with adjusted dimensions
soundcloud_plot <- wordcloud2(sound_freq, size = 3, minRotation = -pi/6, maxRotation = pi/6, rotateRatio = 0.5, gridSize= 3)
soundcloud_plot
# save as html link, then export as PDF

#### Quantify the presence of sounds refering to speech (e.g. talk, speech): ----
# Extract text within brackets and tokenize
data_tokens_sounds <- data_sounds %>%
  mutate(Final_translation = tolower(Final_translation),  # Convert text to lowercase
         words_in_brackets = str_extract_all(Final_translation, "\\[([^]]+)\\]|\\(([^)]+)\\)")) %>%  # Extract text within [] or ()
  unnest(words_in_brackets) %>%
  mutate(words_in_brackets = str_replace_all(words_in_brackets, "[\\[\\]()]", ""),  # Remove brackets
         tokenized_words = str_split(words_in_brackets, "\\s+")) %>%  # Split phrases into individual words
  unnest(tokenized_words) %>%
  filter(tokenized_words != "") %>%  # Remove rows with empty cells
  select(Identifier, tokenized_words)  # Select the relevant columns

# Filter for words of interest
words_of_interest <- c("talk", "talks", "talking", "speak", "speaks", "speaking", "speech", "speeches", "voice", "voices")
sounds_music <- c("music", "musics", "musical", "sing", "singer", "singing", "sings", "song", "rhythm", "rhythmic")
screams_sounds <- c("scream", "screams", "screaming")

# Count occurrences of each word of interest
word_counts <- data_tokens_sounds %>%
  filter(tokenized_words %in% words_of_interest) %>%
  count(tokenized_words, name = "count")

# Calculate the total number of tokenized words
total_tokenized_words <- nrow(data_tokens_sounds)

# Calculate the percentage for each word of interest
word_counts <- word_counts %>%
  mutate(percentage = (count / total_tokenized_words) * 100)

# Calculate total count and total percentage
total_count <- sum(word_counts$count)
total_percentage <- (total_count / total_tokenized_words) * 100

# Add a row for the total count and total percentage
total_row <- tibble(tokenized_words = "Total", count = total_count, percentage = total_percentage)

# Bind the total row to the word_counts dataframe
word_counts <- bind_rows(word_counts, total_row)

# Word counts with percentages
print(word_counts)

# Write the summary table to an Excel file
#write.xlsx(word_counts, "scream sounds.xlsx", rowNames = FALSE)


### Test if word frequency follows Zipf's law ----
# Load data
word_data <- read.csv("list of tokenised word frequency.csv")

# Rank words by frequency
word_data <- word_data[order(-word_data$Freq), ]
word_data$Rank <- seq.int(nrow(word_data))

# Plot data on a log-log scale
ggplot(word_data, aes(x = Rank, y = Freq)) +
  geom_point() +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "Word Frequency vs Rank (Log-Log Scale)",
       x = "Rank",
       y = "Frequency") +
  theme_minimal()

# Fit a Zipfian distribution
# Define a function for the Zipf's law model
zipf_law <- function(rank, c) {
  return(c / rank)
}

# Fit the model
fit <- nls(Freq ~ zipf_law(Rank, c), data = word_data, start = list(c = max(word_data$Freq)))

# Get the fitted value of c
c_fit <- coef(fit)[1]

# Plot fitted Zipfian distribution:
ggplot(word_data, aes(x = Rank, y = Freq)) +
  geom_point() +
  stat_function(fun = function(x) c_fit / x, color = "red") +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "Word rank (log)",
       y = "Word frequency (log)") +
  theme_minimal()+
  theme_bw() +
  My_label_2

# Plot fitted Zipfian distribution:
Zipf1Plot <- ggplot(word_data, aes(x = Rank, y = Freq)) +
  geom_point() +
  stat_function(fun = function(x) c_fit / x, color = "red") +
  scale_x_log10() +
  scale_y_log10(limits = c(1, NA)) +  # Set the lower limit to 1
  labs(x = "Word rank (log)",
       y = "Word frequency (log)") +
  theme_minimal() +
  theme_bw() +
  My_label_2
Zipf1Plot

# Test if the observed frequencies differ from expected frequencies under Zipf's law
# Define observed frequencies
observed <- word_data$Freq

# Create expected frequencies using Zipf's law
expected <- c_fit / word_data$Rank
expected <- expected / sum(expected) * sum(observed)  # Normalize to match observed counts

# Number of unique words (categories)
num_categories <- nrow(word_data)

# Degrees of freedom calculation
df <- num_categories - 1 - 1  # Subtract 1 for the total number of categories and 1 for the estimated parameter

# Now use this df in your interpretation of the chi-square test
cat("Degrees of Freedom: ", df, "\n")

# Chi-Square Goodness of Fit Test
chi_square_test <- chisq.test(observed, p = expected, rescale.p = TRUE)

cat("Chi-Square Test Statistic: ", chi_square_test$statistic, "\n")
cat("P-value: ", chi_square_test$p.value, "\n")


### Zipf's Law of Abbreviation ----

# Calculate word length (number of characters)
word_data$Word_Length <- nchar(word_data$Word)
# Plot word length vs. frequency
Zipf2plot<-ggplot(word_data, aes(x = Freq, y = Word_Length)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +  # Add a linear regression line
  scale_x_log10() +  # Log scale for frequency
  labs(x = "Word frequency (log)",
       y = "Word length") +
  theme_minimal() +
  theme_bw() +
  My_label_2
Zipf2plot
# Perform Spearman's rank correlation test
cor_test <- cor.test(word_data$Freq, word_data$Word_Length, method = "spearman")

# Print correlation coefficient and p-value
cat("Spearman's rank correlation coefficient: ", cor_test$estimate, "\n")
cat("P-value: ", cor_test$p.value, "\n")

# Combine two plots
ggarrange(Zipf1Plot, Zipf2plot, ncol = 2, nrow = 1, labels = c("A)", "B)"), font.label = list(size = 16, color = "red", face = "bold", family = "Times"), label.y = c(1, 1), label.x = c(0.05, 0.05)) 
# Export 750x550


### Intercommunity pant hoots ----
#### Sentiment analysis: ----
# Read the Excel file containing transcribed words from Whisper
dataIC <- read_excel("Data for AI analyses.xlsx", sheet = "IC pant roars")

# Obtain tokenised words
# Replace parentheses, square brackets, specified punctuation, commas, quotation marks, specific hyphen, and the specific character with spaces. 
# Split phrases into individual words
data_tokensIC <- dataIC %>%
  mutate(Final_translation = tolower(Final_translation),  # Convert text to lowercase
         Final_translation = str_replace_all(Final_translation, "[\\[\\]()!,.?…,\"'‘’“”–\\-\\u200B]", " "),  # Replace specified characters with spaces
         Final_translation = str_replace_all(Final_translation, "\\b\\*|\\*\\b", "" ), # Remove asterisks attached to words
         tokenized_words = str_split(Final_translation, pattern = "\\s+")) %>%
  unnest(tokenized_words) %>%
  filter(tokenized_words != "") %>%  # Remove rows with empty cells
  select(Identifier, Final_translation, tokenized_words)

# Ouptut: 61 tokenised words (all "non-word" sounds excluded)
# Check tokenised words
View(data_tokensIC)
write.csv(data_tokensIC, file = "Tokenised words IC pant hoots.csv")

# Number of unique words: 
dataICw <- read_excel("Tokenised words IC pant hoots.xlsx")

# Extract the 'tokenized_words' column
tokenized_wordsW <- dataICw$tokenized_words

# Split words (assuming space-separated words)
words_list <- unlist(strsplit(tokenized_wordsW, " "))

# Get unique words
unique_words <- unique(words_list)

# Count the number of unique words
num_unique_words <- length(unique_words)

# Print the result
cat("Number of unique words:", num_unique_words, "\n")

#### BING lexicon (n = 6786 words): ----
# Load the sentiment lexicon
data("sentiments")

# Export it to inspect
bing_lexicon <- get_sentiments("bing")
#write.xlsx(bing_lexicon, "BING lexicon.xlsx")

# Perform sentiment analysis for each Identifier and tokenized word
sentiment_scores_bing1_IC <- data_tokensIC %>%
  inner_join(get_sentiments("bing"), by = c("tokenized_words" = "word")) %>%
  group_by(Identifier, tokenized_words) %>%
  summarise(avg_sentiment_score = mean(sentiment == "positive") - mean(sentiment == "negative"))

# View sentiment_scores
View(sentiment_scores_bing1_IC) #

# Calculate the number of tokenized words scored
num_tokenized_wordsIC <- data_tokensIC %>% 
  summarise(total_tokenized_words = n())

# Visualise sentiment score per each word. Plot sentiment scores
ggplot(sentiment_scores_bing1_IC, aes(x = avg_sentiment_score)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Sentiment Scores",
       x = "Sentiment Score",
       y = "Frequency") +
  theme_minimal()

# Write tokenised words with bing sentiment score in a csv file
# write.csv(sentiment_scores_bing1_IC, file = "sentiment scores BING IC.csv")

#### AFINN lexicon (n = 2477 words): ----
library("textdata")
afinn_lexicon <- get_sentiments("afinn")

# Export it to inspect:
# write.xlsx(afinn_lexicon, "AFINN lexicon.xlsx")

# Merge tokenized data with AFINN lexicon and include tokenized words
data_sentiment_afinnIC <- data_tokensIC %>%
  left_join(afinn_lexicon, by = c("tokenized_words" = "word")) %>%
  group_by(Identifier, tokenized_words) %>%
  summarise(sentiment_afinn = sum(value, na.rm = TRUE))

# View the sentiment scores
View(data_sentiment_afinnIC)

# Write tokenised words with afinn sentiment score in a csv file:
# write.csv(data_sentiment_afinnIC, file = "sentiment scores AFINN IC.csv")

# Chi-square to compare normal pant hoots and IC pant roars outputs: ----
# Word Translations
translation_data <- matrix(c(91, 306, 13, 89), nrow = 2, byrow = TRUE)
colnames(translation_data) <- c("Translated", "Not Translated")
rownames(translation_data) <- c("Dataset 1", "Dataset 2")
chisq.test(translation_data)
# X-squared = 4.4961, p-value = 0.03397


