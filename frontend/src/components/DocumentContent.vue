<template>
    <v-sheet class="pa-4">
        <h1 class="headline">
            <template v-for="(sentence, sentenceIndex) in title">
                <Sentence :model="sentence" :braceActive="braceActive" :highlightActive="ui.keyClusterProperties.selectedClusters" />
            </template>
        </h1>
        <p v-if="abstract" class="subheading">
            <template v-for="(sentence, sentenceIndex) in abstract">
                <Sentence :model="sentence" :braceActive="braceActive" :highlightActive="ui.keyClusterProperties.selectedClusters" />
            </template>
        </p>
        <p class=".body-1 mb-0">
            <template v-for="(sentence, sentenceIndex) in text">
                <Sentence :model="sentence" :braceActive="braceActive" :highlightActive="ui.keyClusterProperties.selectedClusters" />
            </template>
        </p>
    </v-sheet>
</template>

<script>
import { mapState, mapGetters } from 'vuex';
import Sentence from './Sentence.vue';

export default {
    components: {
        Sentence,
    },
    computed: {
        ...mapState('document', [
            'document',
            'ui',
        ]),
        ...mapGetters('document', [
            'selectedRun',
        ]),
        title() {
            return this.selectedRun.title;
        },
        abstract() {
            return this.selectedRun.abstract;
        },
        text() {
            return this.selectedRun.text;
        },
        braceActive() {
            return [this.ui.commonProperties.showCandidateKeyphrases, this.ui.commonProperties.showSelectedKeyphrases];
        },
    },
};
</script>

<style>

</style>
