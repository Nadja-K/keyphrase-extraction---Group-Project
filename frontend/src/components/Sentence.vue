<template>
    <span>
        <template v-for="(phrase, phraseIndex) in model">
            <span v-if="needsSpace(phrase.words[0].word, phraseIndex)">{{ ' ' }}</span>
            <span :style="getPhraseStyle(phrase)">
                <template v-for="(wordModel, wordIndex) in phrase.words">
                    <v-tooltip bottom>
                        <span slot="activator" :style="getWordStyle(wordModel)">{{ addSpace(wordModel.word, wordIndex) }}</span>
                        <span>{{ getTooltip(wordModel)}}</span>
                    </v-tooltip>
                </template>
            </span>
        </template>
        <span>{{ ' ' }}</span>
    </span>
</template>

<script>
import colors from 'vuetify/es5/util/colors';
import palette from '../palette';

export default {
    props: {
        model: Array,
        tooltipProps: {
            type: Array,
            default() {
                return ['pos', 'cluster', 'weight'];
            },
        },
        braceActive: {
            type: Array,
            default() {
                return [true, true];
            },
        },
        highlightActive: {
            type: Array,
            default() {
                return [...Array(65).keys()];
            },
        },
        phraseStyle: {
            type: Object,
            default() {
                return {
                    brace: {
                        prop: ['candidate_keyphrase', 'candidate_keyphrase_selected'],
                        colorScheme: 'bool',
                    },
                };
            },
        },
        wordStyle: {
            type: Object,
            default() {
                return {
                    bold: 'exemplar_term',
                    highlight: {
                        prop: 'cluster',
                        colorScheme: 'palette',
                    },
                };
            },
        },
    },
    methods: {
        addSpace(word, index) {
            if (this.needsSpace(word, index)) {
                return ` ${word}`;
            }
            return word;
        },
        needsSpace(word, index) {
            return index > 0 && !['.', ',', ':', ';', '"'].includes(word);
        },
        getTooltip(wordModel) {
            const values = this.tooltipProps.map(prop => wordModel.properties[prop]).filter(value => value !== undefined);
            return values.join('; ');
        },
        getPhraseStyle(phrase) {
            return this.getStyle(phrase.properties, this.phraseStyle);
        },
        getWordStyle(wordModel) {
            return this.getStyle(wordModel.properties, this.wordStyle);
        },
        getStyle(model, styleOptions) {
            let style = {};

            if (!model) {
                return style;
            }

            if (styleOptions.bold) {
                if (model[styleOptions.bold]) {
                    style = {
                        fontWeight: 900,
                    };
                }
            }

            if (styleOptions.brace) {
                const color = this.getColor(styleOptions.brace.colorScheme, styleOptions.brace.prop, model, this.braceActive);
                style = {
                    ...style,
                    backgroundImage: `linear-gradient(${color}, ${color}), linear-gradient(${color}, ${color})`,
                    backgroundRepeat: 'no-repeat',
                    backgroundSize: '3px 25%',
                    backgroundPosition: 'bottom left, bottom right',
                    borderBottom: `solid ${color} 2px`,
                    verticalAlign: 'top',
                    padding: '2px 4px',
                };
            }

            if (styleOptions.highlight) {
                const color = this.getColor(styleOptions.highlight.colorScheme, styleOptions.highlight.prop, model, this.highlightActive);
                style = {
                    ...style,
                    backgroundColor: color,
                };
            }

            return style;
        },
        getColor(option, prop, model, active) {
            let color = 'transparent';

            switch (option) {
            case 'palette': {
                const value = model[prop];
                if (value && active.includes(value)) {
                    color = `#${palette[value]}`;
                }
                break;
            }
            case 'bool': {
                if (active[0] && model[prop[0]]) {
                    color = colors.grey.base;
                }
                if (active[1] && model[prop[1]]) {
                    color = colors.red.lighten2;
                }
                break;
            }
            default:
                break;
            }

            return color;
        },
    },
};
</script>

<style>
</style>
